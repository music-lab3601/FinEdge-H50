"""Isolated local queue32 dashboard; no board access in HTTP handlers."""
from pathlib import Path
import argparse,csv,http.cookiejar,io,json,mimetypes,secrets,sys,threading,time,traceback,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
import urllib.request
U=Path(__file__).resolve().parent;G=U.parents[1];P=G.parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(G))
from integration.common import read
from storage import atomic_json
from integration.resource_queue import ResourceLock,Busy
from telemetry import Telemetry,read_json
from runner import run
from evidence import Evidence
from dispatch_timing import checked_speed,wall_duration_ns

class App:
    def __init__(self,review=None):
        self.guard=threading.RLock();self.worker=None;self.closed=threading.Event();self.closing=False;self.phase='idle'
        self.message='PS串口负责管理，网线连接PL口；启动后自动软件重启并建立新会话。'
        self.error=None;self.directory=None;self.telemetry=None;self.review=bool(review);self.round=0
        self.config=read(U/'config.json');self.last=None;self.evidence=None;self.dispatch_speed=1;self.clock_metadata=None;self.final_telemetry=None
        if review:
            self.directory=Path(review).resolve();self.telemetry=Telemetry(self.directory,self.config['reference'])
            self.phase='review';self.message='真实板测记录复盘，当前未连接开发板。'
        else:
            current=read_json(U/'artifacts/CURRENT.json',dict(directory=self.config.get('initial_record',self.config['seed_run'])))
            if current:
                latest=Path(current['directory']).resolve()
                completed=read_json(latest/'showcase_result.json',{}) if latest.is_relative_to((U/'artifacts').resolve()) or latest in [Path(self.config[k]).resolve() for k in ('seed_run','initial_record') if self.config.get(k)] else {}
                if completed.get('status')=='passed' and completed.get('cleanup')=='software_reboot_verified':
                    self.directory=latest;self.telemetry=Telemetry(latest,self.config['reference']);self.phase='finished'
                    self.message='已载入上次真实板测结果；当前未启动板卡任务。点击“启动新一轮”开始新的行情回放。'
        if self.telemetry:
            self.evidence=Evidence(self.directory,self.config,self.telemetry)
            self.dispatch_speed=checked_speed(read_json(self.directory/'run_manifest.json',{}).get('speed',1))
    def note(self,text):
        with self.guard:self.message=text
    def clock_state(self):
        # A clock request must not queue behind full telemetry/evaluation scans.
        # Only immutable session metadata and the source's actual QPC anchor
        # are read; event/ACK/prediction counts are never extrapolated here.
        phase=self.phase;directory=self.directory;telemetry=self.telemetry;epoch=self.round
        if phase!='running' or directory is None or telemetry is None:
            return dict(mode='snapshot',phase=phase,round=epoch)
        key=(str(directory),epoch)
        cache=self.clock_metadata
        if cache is None or cache[0]!=key:
            manifest=read_json(directory/'run_manifest.json')
            anchor=read_json(directory/'sender/source/anchor.json')
            if not manifest or not anchor:
                return dict(mode='preparing',round=epoch)
            if anchor['session']!=manifest['session'] or anchor['first']!=manifest['fixture']['first_raw_ordinal']:
                raise ValueError('Display clock source/session mismatch')
            speed=checked_speed(manifest['speed']);fixture=manifest['fixture']
            metadata=dict(round=epoch,directory=str(directory),session=str(manifest['session']),
                origin_market_unix_ms=int(telemetry.rows[0]['event_ns'])/1e6,
                anchor_ns=anchor['anchor_ns'],wall_duration_ms=wall_duration_ns(fixture['duration_ns'],speed)/1e6,
                speed=speed,utc_offset_minutes=-300)
            cache=(key,metadata);self.clock_metadata=cache
        if self.round!=epoch or self.directory!=directory or self.phase!='running':
            return dict(mode='snapshot',phase=self.phase,round=self.round)
        metadata=dict(cache[1]);anchor_ns=metadata.pop('anchor_ns')
        return dict(mode='live',elapsed_wall_ms=(time.perf_counter_ns()-anchor_ns)/1e6,**metadata)
    def state(self):
        with self.guard:
            phase=self.phase;message=self.message;error=self.error
            if self.directory and phase=='running':
                p=read_json(self.directory/'ui_progress.json',{})
                message=p.get('message',message)
            try:
                if self.telemetry:
                    if phase in ('finished','review'):
                        if self.final_telemetry is None:self.final_telemetry=self.telemetry.snapshot(final=True)
                        self.last=self.final_telemetry
                    else:self.last=self.telemetry.snapshot()
                else:self.last=None
            except Exception as exc:error='显示核验失败：'+str(exc)
            if phase=='running' and self.last and p.get('stage')=='running':
                if not self.last.get('raw_events'):
                    message='正在恢复订单前缀：'+format(self.last.get('bootstrap_committed',0),',')+' / '+format(self.last.get('bootstrap_target',0),',')+f' 条；随后按{self.dispatch_speed}倍历史事件时钟派发测量片段。'
                else:message=f'正在按{self.dispatch_speed}倍历史事件时钟派发原始MBO；FPGA在线处理，预测经PL网口返回。'
            try:evidence=self.evidence.snapshot(final=phase in ('finished','review')) if self.evidence else {}
            except Exception as exc:evidence={};error='源与窗口证据核验失败：'+str(exc)
            return dict(schema='finedge-integer-stream-gui-v5',dispatch_speed=self.dispatch_speed,phase=phase,message=message,error=error,review=self.review,
                        busy=bool(self.worker and self.worker.is_alive()),closing=self.closing,round=self.round,
                        telemetry=self.last,directory=str(self.directory) if self.directory else None,
                        book_source='同源重建参考，仅用于显示与方向兑现，不作为板端输入',CPU_reference=False,Host_simulation=False,
                        pipeline=dict(name=self.config['display_name'],preprocessing=self.config['preprocessing_version'],transport=self.config['transport_version'],firmware_sha256=self.config['BOOT_sha256'],record_firmware_sha256=read_json(self.directory/'run_manifest.json',{}).get('loaded_firmware',{}).get('boot_sha256') if self.directory else None),**evidence)
    def action(self,name,payload):
        with self.guard:
            if name=='close':self.closing=True;self.message='当前片段处理并保存后关闭服务。';return self.state()
            if name in ('track','replay','jump'):
                self.state()
                if not self.evidence or not self.evidence.manifest:raise ValueError('当前没有可查看的运行记录。')
                if name=='track':self.evidence.choose(payload.get('opportunity'))
                else:
                    if self.phase not in ('finished','review') or (self.worker and self.worker.is_alive()):raise ValueError('真实行情运行中不暂停时钟；本轮保存后可复盘和步进。')
                    if name=='jump':self.evidence.jump(payload.get('stage'))
                    else:self.evidence.clock.command(payload.get('operation'),self.evidence.bounds()[1],**{k:v for k,v in payload.items() if k!='operation'})
                return self.state()
            if name!='start':raise ValueError('未知操作。')
            if self.review:raise ValueError('复盘窗口不连接板卡，请使用启动入口。')
            if self.closing or (self.worker and self.worker.is_alive()):raise ValueError('当前片段尚未结束。')
            minutes=payload.get('minutes',5);slots=payload.get('slots',32);age=payload.get('max_age_us',2000);speed=checked_speed(payload.get('speed',1))
            if type(minutes)!=int or minutes not in (1,2,5) or type(slots)!=int or slots not in (2,4,8,16,32) or type(age)!=int or not 0<=age<=10000:raise ValueError('无效运行配置：时长仅支持1、2、5分钟。')
            self.dispatch_speed=speed;self.round+=1;self.phase='running';self.error=None;self.telemetry=None;self.last=None;self.directory=None;self.evidence=None;self.clock_metadata=None;self.final_telemetry=None
            self.message='正在准备全硬件通路。'
            def directory(d):
                with self.guard:self.directory=d;self.telemetry=Telemetry(d,self.config['reference']);self.evidence=Evidence(d,self.config,self.telemetry)
            def work():
                try:
                    d=run(minutes,slots,age,on_directory=directory,note=self.note,speed=speed)
                    with self.guard:
                        final=self.telemetry.snapshot(final=True)
                        evaluation=dict(schema='finedge-queue32-direction-v1',summary=final['evaluation'],data_source=self.config['reference'],future_prices_hidden_until_committed=True)
                        atomic_json(d/'direction_evaluation.json',evaluation)
                        self.phase='finished';self.message='本轮完成，逐窗结果已核验并保存，板卡会话已通过软件重启回收。'
                except Exception as exc:
                    with self.guard:self.error=str(exc);self.phase='failed';self.message='本轮失败，已有数据和现场已保存；请查看下方错误与日志位置。'
            self.worker=threading.Thread(target=work,name='queue32-run',daemon=False);self.worker.start()
        return self.state()
    def export(self):
        with self.guard:
            if self.phase not in ('finished','review'):raise ValueError('本轮结束后可导出完整预测日志。')
            self.telemetry.snapshot(final=True);rows=self.telemetry.last_predictions
            text=io.StringIO();writer=csv.writer(text)
            writer.writerow(['机会编号','原始记录编号','预测类别_0跌1平2涨','实际类别','方向命中','H50参考及时','兑现状态','中间价变化_USD','端到端_ms','NPU启动等待_us','板内调用链_us'])
            for r in rows:writer.writerow([r['opportunity'],r['ordinal'],r['class_index'],r['actual_class'],r['hit'],r['H50_timely'],r['settlement'],r['mid_change'],r['age_ms'],r['npu_us'],r['service_us']])
            return text.getvalue().encode('utf-8-sig')

def make_server(app,port=0):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,data,content='application/json; charset=utf-8',cookie=False):
            self.send_response(code);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            # Browser cookies are scoped by host/path, never by TCP port.
            # Keep live and review services from replacing each other's token.
            if cookie:self.send_header('Set-Cookie','FE_QUEUE_'+str(self.server.server_port)+'='+token+'; HttpOnly; SameSite=Strict; Path=/')
            if content=='text/csv; charset=utf-8':self.send_header('Content-Disposition','attachment; filename="FinEdge_predictions.csv"')
            self.end_headers();self.wfile.write(data)
        def auth(self):
            expected='127.0.0.1:'+str(self.server.server_port)
            return self.headers.get('Host')==expected and any(x.strip()=='FE_QUEUE_'+str(self.server.server_port)+'='+token for x in self.headers.get('Cookie','').split(';'))
        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path.startswith('/api/'):
                    if not self.auth():return self.reply(403,b'{}')
                    if path=='/api/clock':return self.reply(200,json.dumps(app.clock_state(),allow_nan=False).encode())
                    if path=='/api/state':return self.reply(200,json.dumps(app.state(),ensure_ascii=False,allow_nan=False).encode())
                    if path=='/api/export':return self.reply(200,app.export(),'text/csv; charset=utf-8')
                    return self.reply(404,b'{}')
                relative='index.html' if path in ('/','/source','/track') else path.lstrip('/')
                target=(U/'web'/relative).resolve()
                if not target.is_relative_to((U/'web').resolve()) or not target.is_file():return self.reply(404,b'Not found','text/plain')
                self.reply(200,target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream',cookie=path in ('/','/source','/track'))
            except (ValueError,OSError) as exc:self.reply(400,json.dumps(dict(error=str(exc)),ensure_ascii=False).encode())
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            origin='http://127.0.0.1:'+str(self.server.server_port)
            if not self.auth() or self.headers.get('Origin')!=origin:return self.reply(403,b'{}')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('无效请求长度。')
                payload=json.loads(self.rfile.read(length));action=urlsplit(self.path).path.removeprefix('/api/')
                if action not in ('start','close','track','replay','jump'):raise ValueError('未知操作。')
                self.reply(200,json.dumps(app.action(action,payload),ensure_ascii=False,allow_nan=False).encode())
                if action=='close':
                    # Return the complete response before the main thread can
                    # close the HTTP server, including expensive final stats.
                    self.wfile.flush();app.closed.set()
            except (ValueError,TypeError) as exc:self.reply(400,json.dumps(dict(error=str(exc)),ensure_ascii=False).encode())
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

def existing_url():
    path=U/'artifacts/live_url.txt'
    if not path.exists():return None
    url=path.read_text().strip();p=urlsplit(url)
    if p.scheme!='http' or p.hostname!='127.0.0.1' or not p.port or p.path not in ('','/') or p.query or p.fragment:return None
    try:
        jar=http.cookiejar.CookieJar();opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.open(url,timeout=2).read();v=json.load(opener.open(url+'/api/state',timeout=2))
        return url if v.get('schema')=='finedge-integer-stream-gui-v5' else None
    except (OSError,ValueError):return None

def print_url(url):
    # pythonw launched from CMD may inherit a non-null but invalid stdout.
    # The desktop entry communicates via URL files and the browser instead.
    try:
        if sys.stdout:print(url,flush=True)
    except OSError:pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path);ap.add_argument('--review-latest',action='store_true');ap.add_argument('--no-browser',action='store_true');ap.add_argument('--port',type=int,default=0);ap.add_argument('--view',choices=('source','track','overview'),default='overview');a=ap.parse_args()
    if a.review_latest:
        current=read_json(U/'artifacts/CURRENT.json')
        if current:a.review=Path(current['directory'])
        else:a.review=Path(read(U/'config.json')['seed_run'])
    lock=None
    if not a.review:
        lock=ResourceLock(P/'artifacts/launchers','launcher')
        try:lock.__enter__()
        except Busy:
            url=existing_url()
            if url:
                if not a.no_browser:webbrowser.open(url+('/'+a.view if a.view!='overview' else '/'))
                print_url(url);return
            raise RuntimeError('已有演示窗口占用启动器，请先结束其窗口服务。')
    app=server=None
    try:
        app=App(a.review);server=make_server(app,a.port);url='http://127.0.0.1:'+str(server.server_port)
        (U/'artifacts'/('review_url.txt' if a.review else 'live_url.txt')).write_text(url,encoding='utf-8')
        print_url(url)
        if not a.no_browser:webbrowser.open(url+('/'+a.view if a.view!='overview' else '/'))
        def shutdown():
            app.closed.wait()
            if app.worker:app.worker.join()
            server.shutdown()
        threading.Thread(target=shutdown,daemon=True).start();server.serve_forever(poll_interval=.2)
    finally:
        if app and app.worker:app.worker.join()
        if server:server.server_close()
        if lock:lock.__exit__(None,None,None)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        with (U/'artifacts/launch_errors.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+'\n'+traceback.format_exc()+'\n')
        reported=False
        if sys.stderr:
            try:print(str(exc),file=sys.stderr);reported=True
            except OSError:pass
        if not reported:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,str(exc)+'\n\n详见演示目录artifacts/launch_errors.log','FinEdge 启动提示',0x10)
