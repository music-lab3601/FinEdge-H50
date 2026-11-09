"""Local financial dashboard adapter. Inference remains in run_showcase.

Only bounded display snapshots are polled. Opening/refreshing the page never
opens a device session. Review mode reads recorded evidence and cannot start one.
"""
from __future__ import annotations

from copy import deepcopy
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import mimetypes
import secrets
import shutil
import threading
import time
import traceback

from experiments.reference.runtime.demo_rate import validate_rate
from experiments.reference.runtime.fpga_streaming import LatestDisplay
from shared.model_paths import ROOT
from experiments.reference.runtime.showcase_replay import BACKEND, export_showcase_csv, run_showcase
from experiments.reference.runtime.showcase_monitor import MonitoringDisplay

ASSETS = Path(__file__).with_name('web')


def project_snapshot(snapshot):
    """Keep browser payload bounded; no tokens, tensors or private contracts."""
    if not snapshot:
        return None
    keys = ('stats', 'evaluation', 'market', 'series', 'recent_predictions',
            'latest_prediction_id', 'latest_resolved_id', 'expected_windows',
            'elapsed_seconds', 'observed_windows_per_second', 'phase', 'status', 'error', 'monitor')
    result = {key: deepcopy(snapshot[key]) for key in keys if key in snapshot}
    rows = result.get('recent_predictions', [])
    result['book'] = rows[-1].get('book') if rows else None
    for row in rows:
        row.pop('book', None)
    return result


def recorded_snapshot(directory):
    """Render the already captured real run; never calculate new predictions."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    mailbox = LatestDisplay()
    view = MonitoringDisplay(mailbox, clock=lambda: 0)
    binary_path = directory/'board_native_predictions.bin'
    binary = binary_path.open('rb') if binary_path.is_file() else None
    try:
        for name in ('market', 'predictions', 'outcomes'):
            with (directory/(name+'.jsonl')).open(encoding='utf-8') as source:
                for line in source:
                    event = json.loads(line)
                    # Recover existing stage evidence without modifying original logs.
                    if binary and name == 'predictions' and not event.get('startup_included', False):
                        from experiments.reference.runtime.native_showcase import PACKET, decode_packet
                        packet = binary.read(PACKET.size)
                        if len(packet) != PACKET.size:
                            raise ValueError('Recorded native timing packet missing')
                        native = decode_packet(packet)
                        output = event['outputs'][BACKEND]
                        if (native['window_id'] != event['window_id'] or native['logits'] != output['logits']
                                or native['board_pipeline_ns'] != event['board_pipeline_ns']):
                            raise ValueError('Recorded timing/result identity mismatch')
                        event['stages_ns'] = native['stages_ns']
                    view.consume(event)
        if binary and binary.read(1):
            raise ValueError('Unexpected extra recorded timing packets')
    finally:
        if binary:
            binary.close()
    view.consume(dict(type='finished', summary=manifest))
    return view.snapshot(), manifest


class WebShowcase:
    def __init__(self, *, review_directory=None, setup=None, runner=run_showcase):
        self.guard = threading.RLock()
        self.mailbox = LatestDisplay()
        self.paused = threading.Event()
        self.cancelled = threading.Event()
        self.closed = threading.Event()
        self.worker = None
        self.runner = runner
        self.setup = setup
        self.rate = 20.0
        self.model_mode = 'fused'
        self.phase = 'idle'
        self.message = '选择模型与回放速度，启动后连接悟净开发板。'
        self.error = None
        self.session = None
        self.directory = None
        self.snapshot = None
        self.summary = None
        self.revision = 0
        self.export_lock = threading.Lock()
        self.review = review_directory is not None
        self.events = []
        if self.review:
            self.snapshot, self.summary = recorded_snapshot(review_directory)
            self.directory = Path(review_directory)
            self.phase = 'review'
            self.model_mode = self.summary.get('model_mode', 'fused')
            self.message = '真实板测记录复盘 · 当前未连接开发板，显示已保存的预测与耗时。'
        self.note(self.message)

    def note(self, message):
        with self.guard:
            self.message = str(message)
            self.events.append({'time': time.strftime('%H:%M:%S'), 'message': self.message})
            self.events = self.events[-40:]
            self.revision += 1

    def state(self):
        with self.guard:
            fresh = self.mailbox.take()
            if fresh:
                self.snapshot = fresh
                self.revision += 1
                if fresh.get('directory'):
                    self.directory = Path(fresh['directory'])
            snapshot = self.snapshot
            phase = self.phase
            if phase == 'running' and snapshot:
                phase = snapshot.get('phase', phase)
            return dict(schema='finedge-web-v1', revision=self.revision,
                phase=phase, busy=bool(self.worker and self.worker.is_alive()),
                pause_requested=self.paused.is_set(), stop_requested=self.cancelled.is_set(),
                review=self.review, rate=self.rate, model_mode=self.model_mode,
                message=self.message, error=self.error, events=list(self.events),
                data_label='2025-12-15 · AAPL · 原始 MBO 运行时重建',
                validation_aligned=False, snapshot=project_snapshot(snapshot),
                log_available=bool(self.directory and (self.directory/'predictions.jsonl').exists()))

    def start(self, payload):
        with self.guard:
            if self.review:
                raise ValueError('复盘模式不连接开发板，请使用正式演示入口。')
            if self.closed.is_set():
                raise ValueError('窗口服务已关闭。')
            if self.worker and self.worker.is_alive():
                raise ValueError('已有任务正在运行。')
            if self.session is not None or self.phase != 'idle':
                raise ValueError('本次会话已使用；请结束此窗口服务后重新打开演示入口。')
            mode = payload.get('model', 'fused')
            if mode not in ('fused', 'strict'):
                raise ValueError('未知模型模式。')
            rate = validate_rate(payload.get('rate', 20))
            checkpoint = payload.get('checkpoint', 10000)
            if type(checkpoint) is not int or not 1 <= checkpoint <= 100000:
                raise ValueError('首段窗口数量应在 1～100000 之间。')
            self.rate, self.model_mode = rate, mode
            self.phase = 'preparing'
            self.note('准备模型与行情来源；随后软件重启开发板并建立新会话。')
            self.worker = threading.Thread(target=self._work, args=(mode, checkpoint),
                                           name='finedge-web-inference', daemon=False)
            self.worker.start()

    def _work(self, mode, checkpoint):
        try:
            if self.setup:
                session = self.setup(mode, self.note)
            else:
                from experiments.reference.runtime.native_showcase import prepare_native_showcase, deploy_native_showcase
                session = deploy_native_showcase(prepare_native_showcase(mode), self.note)
            with self.guard:
                self.session = session
                self.phase = 'running'
            self.note('设备已就绪，正在加载行情并初始化 FPGA＋NPU 通路。')
            result = self.runner(session, display=self.mailbox, paused=self.paused,
                cancelled=self.cancelled, get_rate=lambda: self.rate, max_windows=checkpoint,
                display_factory=MonitoringDisplay)
            with self.guard:
                self.summary = result
                self.directory = Path(result['directory'])
                self.phase = result['status']
                self.error = result.get('error')
            self.note({'completed': '行情回放完成，结果已保存。',
                       'partial': '已停止并保存，可以查看与导出结果。',
                       'failed': '运行故障，已保留日志；本次会话不可重试。'}.get(self.phase, self.phase))
            # Record UI identity alongside the run, without changing frozen board evidence.
            files = [Path(__file__), Path(__file__).with_name('showcase_monitor.py'),
                     *ASSETS.glob('*.js'), *ASSETS.glob('*.css'), *ASSETS.glob('*.html')]
            identity = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files}
            (self.directory/'web_ui_identity.json').write_text(json.dumps(dict(
                schema='finedge-web-v1', chart_version='5.2.1', sources=identity,
                cpu_reference_enabled=False, host_simulation_enabled=False), indent=2)+'\n', encoding='utf-8')
        except Exception as exc:
            with self.guard:
                self.phase = 'failed'
                self.error = str(exc)
            self.note('启动或运行失败：'+str(exc))
            log = ROOT/'artifacts/web_showcase'
            log.mkdir(parents=True, exist_ok=True)
            (log/('error_'+time.strftime('%Y%m%d_%H%M%S')+'.txt')).write_text(traceback.format_exc(), encoding='utf-8')

    def action(self, name, payload):
        if name == 'start':
            self.start(payload)
        elif name == 'rate':
            with self.guard:
                self.rate = validate_rate(payload['rate'])
            self.note('回放速度已调整为 '+('不限速' if not self.rate else f'{self.rate:g} 窗/秒')+'。')
        elif name in ('pause', 'resume', 'stop'):
            with self.guard:
                if not self.worker or not self.worker.is_alive() or self.phase != 'running':
                    raise ValueError('当前没有可控制的运行任务。')
                if self.cancelled.is_set():
                    raise ValueError('正在停止，等待当前批次结束并保存。')
                if name == 'pause':
                    self.paused.set()
                    self.note('正在等待当前批次完成后暂停…')
                elif name == 'resume':
                    self.paused.clear()
                    self.note('继续回放，沿用当前会话。')
                else:
                    self.cancelled.set()
                    self.paused.clear()
                    self.note('正在停止并保存，请保持开发板和网络连接。')
        elif name == 'close':
            self.cancelled.set()
            self.paused.clear()
            self.closed.set()
            self.note('正在结束窗口服务并保存运行结果。')
        else:
            raise ValueError('未知操作。')
        return self.state()


def make_server(controller, port=0):
    cookie_token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, content_type='application/json; charset=utf-8', *, cookie=False):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, dict) else data
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if cookie:
                self.send_header('Set-Cookie', f'finedge={cookie_token}; HttpOnly; SameSite=Strict; Path=/')
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            try:
                cookies = SimpleCookie(self.headers.get('Cookie', ''))
                return self.headers.get('Host') == self.server.local_host and secrets.compare_digest(
                    cookies['finedge'].value, cookie_token)
            except (KeyError, ValueError):
                return False

        def do_GET(self):
            if self.headers.get('Host') != self.server.local_host:
                return self.send(403, {'error': '仅允许本机窗口访问。'})
            if self.path == '/':
                return self.send(200, (ASSETS/'index.html').read_bytes(), 'text/html; charset=utf-8', cookie=True)
            if not self.authorized():
                return self.send(403, {'error': '请先打开演示首页。'})
            if self.path == '/api/state':
                return self.send(200, controller.state())
            if self.path == '/api/export':
                if not controller.export_lock.acquire(blocking=False):
                    return self.send(409, {'error': '正在导出，请稍候。'})
                try:
                    directory = controller.directory
                    if not directory:
                        return self.send(409, {'error': '尚无可导出的结果。'})
                    limits = {n: (directory/(n+'.jsonl')).stat().st_size for n in ('predictions', 'outcomes')}
                    exports = ROOT/'artifacts/web_showcase/exports'
                    exports.mkdir(parents=True, exist_ok=True)
                    destination = exports/('web_export_'+secrets.token_hex(4)+'.csv')
                    export_showcase_csv(directory, destination, limits=limits)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/csv; charset=utf-8')
                    self.send_header('Content-Disposition', 'attachment; filename="FinEdge_predictions.csv"')
                    self.send_header('Content-Length', str(destination.stat().st_size))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    with destination.open('rb') as source:
                        shutil.copyfileobj(source, self.wfile)
                except (OSError, ValueError) as exc:
                    controller.note('导出未完成：'+str(exc))
                finally:
                    controller.export_lock.release()
                return
            static = {'/app.js': 'app.js', '/app.css': 'app.css',
                '/vendor/lightweight-charts.js': 'vendor/lightweight-charts/lightweight-charts.standalone.production.js',
                '/vendor/NOTICE': 'vendor/lightweight-charts/NOTICE', '/vendor/LICENSE': 'vendor/lightweight-charts/LICENSE'}
            if self.path not in static:
                return self.send(404, {'error': '不存在的页面。'})
            path = ASSETS/static[self.path]
            mime = 'text/javascript' if path.suffix == '.js' else mimetypes.guess_type(path.name)[0] or 'text/plain'
            self.send(200, path.read_bytes(), mime+'; charset=utf-8')

        def do_POST(self):
            origin = 'http://'+self.server.local_host
            if not self.authorized() or self.headers.get('Origin') != origin:
                return self.send(403, {'error': '操作必须来自本机演示窗口。'})
            try:
                if self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('需要 JSON 请求。')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('无效请求长度。')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or not self.path.startswith('/api/'):
                    raise ValueError('无效操作。')
                result = controller.action(self.path.removeprefix('/api/'), payload)
                self.send(200, result)
            except (KeyError, ValueError, TypeError) as exc:
                self.send(400, {'error': str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.local_host = f'127.0.0.1:{server.server_port}'
    return server
