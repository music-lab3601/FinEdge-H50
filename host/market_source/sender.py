"""Same source/control implementation for ARM/PS and FPGA/PL.

Bootstrap is credited before measurement. Measured dispatch is the frozen
native source, with no receiver feedback; the observer only records replies.
"""
from pathlib import Path
import argparse,json,socket,struct,subprocess,sys,threading,time
from shared.paths import work_path
ROOT=work_path('market_source')
from shared.io import PROJECT, read, sha, atomic_json
from contracts.market.finedge_contract import event_packet
from host.telemetry.dispatch_timing import checked_speed, wall_duration_ns
ACK=struct.Struct('<II7Q');OBS=struct.Struct('<9Q')


def pin_host_thread(core):
 if sys.platform=='win32':
  import ctypes
  k=ctypes.WinDLL('kernel32',use_last_error=True)
  k.GetCurrentThread.restype=ctypes.c_void_p
  k.SetThreadAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
  k.SetThreadAffinityMask.restype=ctypes.c_size_t
  if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<core):raise OSError(ctypes.get_last_error(),'Host thread affinity')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',choices=['smoke','normal','busy','controlled','integrated'],required=True)
 ap.add_argument('--host',required=True);ap.add_argument('--local',default='192.168.126.1');ap.add_argument('--session',type=int,required=True)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--functional',action='store_true');ap.add_argument('--speed',type=int,default=1,choices=(1,2,4))
 ap.add_argument('--rate',type=int,default=0);ap.add_argument('--data-port',type=int,default=20001);ap.add_argument('--ack-port',type=int,default=20002)
 ap.add_argument('--release',type=Path);ap.add_argument('--drain-seconds',type=float,default=1.0)
 a=ap.parse_args();checked_speed(a.speed);a.out.mkdir(parents=True,exist_ok=False)
 if a.fixture=='integrated':
  release=read(a.release);fixture=release['fixture'];raw=Path(release['raw_source']);digest=release['raw_source_sha256']
 elif a.fixture=='smoke':
  if not a.functional:raise ValueError('Synthetic smoke is functional only')
  fixture=read(ROOT/'fixtures/smoke/release.json');raw=ROOT/'fixtures/smoke/raw56.bin';digest=fixture['raw_source_sha256']
 elif a.fixture=='controlled':
  release=read(ROOT/'state/fair_controlled_fixture.json');fixture=release['fixture'];raw=Path(release['raw_source']);digest=release['raw_source_sha256']
  if a.functional or a.rate<=0:raise ValueError('Controlled rate requires explicit positive rate, sampled output')
 else:
  release=read(ROOT/'fixtures/r1/release.json');fixture=next(f for f in release['fixtures'] if f['id']==a.fixture)
  raw=PROJECT/'versions/network_fast_v1/artifacts/raw_mbo56.bin';digest=release['raw_source_sha256']
 if not 0<a.session<2**64 or a.speed<=0 or sha(raw)!=digest:raise ValueError('Session/source identity')
 build=read(work_path('native/source_build.json'));source=Path(build['executable'])
 if sha(source)!=build['sha256']:raise ValueError('Isolated visual source changed')
 first=fixture['first_raw_ordinal'];last=fixture['last_raw_ordinal'];prefix=first-1
 result=dict(status='initializing',fixture=fixture,host=a.host,session=a.session,source_binary_sha256=sha(source),raw_source_sha256=digest,
             measured_feedback=False,functional=a.functional,schedule='rate' if a.rate else 'historical',speed=a.speed,rate=a.rate)
 pin_host_thread(0)
 receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,262144)
 receiver.bind((a.local,a.ack_port));receiver.settimeout(.1)
 sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sender.bind((a.local,0));sender.connect((a.host,a.data_port))
 stopped=threading.Event();condition=threading.Condition();observations=[];prediction_packets=[];current=dict(committed=0,ready=False,error=None)
 def observe():
  pin_host_thread(2)
  while not stopped.is_set():
   try:data,peer=receiver.recvfrom(2048)
   except socket.timeout:continue
   stamp=time.perf_counter_ns()
   if peer[0]!=a.host:continue
   if len(data)==192:
    header=struct.unpack_from('<IIQQQ3fIQQII',data)
    if header[0]!=0x32524d46 or header[1]!=2:continue
    if header[2]!=a.session:continue
    prediction_packets.append((stamp,data));continue
   if len(data)!=64:continue
   magic,flags,committed,eligible,windows,batches,error,ticks,session=ACK.unpack(data)
   if magic!=0x31414d46 or session!=a.session:continue
   observations.append((stamp,flags,committed,eligible,windows,batches,error,ticks,session))
   with condition:
    if flags&1 or error:current['error']='Receiver fault '+hex(error)
    elif committed<current['committed']:current['error']='Committed ordinal regressed'
    elif committed>last:current['error']='Committed ordinal beyond frozen end'
    else:
     current['committed']=committed
     if flags&6==6:current['ready']=True
    condition.notify_all()
 thread=threading.Thread(target=observe);thread.start()
 writer_stop=threading.Event();display_errors=[]
 def publish_live():
  # The receive thread only timestamps and appends to RAM. File publication
  # runs separately and cannot backpressure the independently clocked source.
  try:
   with (a.out/'predictions_live.bin').open('xb') as pf,(a.out/'observations_live.bin').open('xb') as af:
    pi=ai=0
    while True:
     pn=len(prediction_packets);an=len(observations)
     pf.write(b''.join(struct.pack('<Q',t)+p for t,p in prediction_packets[pi:pn]))
     af.write(b''.join(OBS.pack(*r) for r in observations[ai:an]));pf.flush();af.flush();pi=pn;ai=an
     if writer_stop.is_set():break
     writer_stop.wait(.05)
  except BaseException as exc:display_errors.append(type(exc).__name__+': '+str(exc))
 live_thread=threading.Thread(target=publish_live);live_thread.start()
 def wait(target,ready=False,timeout=30):
  end=time.monotonic()+timeout
  with condition:
   while not (current['ready'] if ready else current['committed']>=target):
    if current['error']:raise RuntimeError(current['error'])
    remaining=end-time.monotonic()
    if remaining<=0:raise TimeoutError('Receiver progress '+str((target,current['committed'])))
    condition.wait(min(.1,remaining))
   if current['error']:raise RuntimeError(current['error'])
 try:
  wait(0,True,10)
  if current['committed']!=0:raise RuntimeError('Fresh empty session required')
  packet=1;position=1;end=last if a.functional else prefix;begin=time.monotonic()
  with raw.open('rb') as stream:
   while position<=end:
    count=min(24,end-position+1);payload=stream.read(56*count)
    if len(payload)!=56*count:raise ValueError('Raw prefix truncated')
    sender.send(event_packet(payload,a.session,packet,position));packet+=1;position+=count
    if packet==2 or position-1-current['committed']>=96 or position>end:wait(position-1)
    if (position-1)//65536!=(position-1-count)//65536:atomic_json(a.out/'bootstrap_progress.json',dict(sent=position-1,committed=current['committed']))
  result.update(bootstrap_seconds=time.monotonic()-begin,bootstrap_events=end,bootstrap_completed=True)
  atomic_json(a.out/'result.json',result)
  if not a.functional:
   argv=[str(source),'--raw',str(raw),'--first',str(first),'--last',str(last),'--session',str(a.session),
         '--first-packet',str(packet),'--host',a.host,'--port',str(a.data_port),'--out',str(a.out/'source'),
         '--schedule','rate' if a.rate else 'historical','--packetization','single' if a.rate else 'same-ts',
         '--speed',str(a.speed),'--rate',str(a.rate),'--core','4','--lead-ms','200','--send-buffer-bytes','262144','--live-publications','1','--duration-ns',str(wall_duration_ns(fixture.get('duration_ns',0),a.speed))]
   result['measured_argv']=argv;atomic_json(a.out/'result.json',result)
   with (a.out/'source_console.log').open('xb') as log:
    # The observer may record a consumer failure, but cannot stop or re-time
    # this finite native dispatch. END/cleanup belongs to the board runner.
    run=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=max(120,wall_duration_ns(fixture.get('duration_ns',0),a.speed)/1e9+60))
   result['source']=read(a.out/'source/source.json')
   if run.returncode:raise RuntimeError('Native source failed; no retry')
   result['last_source_submission_ns']=result['source']['last_send_done_ns']
   wait(last,timeout=5)
  result.update(status='all_events_committed',committed=current['committed'],next_packet_seq=packet)
  time.sleep(a.drain_seconds)
 except BaseException as exc:
  result.update(status='failed',error=type(exc).__name__+': '+str(exc),committed=current['committed'])
 finally:
  stopped.set();thread.join(timeout=2);sender.close();receiver.close();writer_stop.set();live_thread.join(timeout=3)
  with (a.out/'observations.bin').open('xb') as log:
   for row in observations:log.write(OBS.pack(*row))
  with (a.out/'predictions.bin').open('xb') as log:
   for stamp,payload in prediction_packets:log.write(struct.pack('<Q',stamp)+payload)
  result.update(prediction_packets=len(prediction_packets),display_writer_errors=display_errors,prediction_transport='FPGA FMR2 UDP on same PL link',
                event_to_prediction_boundary='PC QPC raw scheduled due -> PC QPC prediction UDP receive; independent of FPGA clocks',
                observer_records=len(observations),receiver_error=current['error'],one_way_end_to_end_ns=None,
                observer_scope='PC receive time and cumulative commit replies; includes report cadence/network lag')
  atomic_json(a.out/'result.json',result)
 print(json.dumps(result),flush=True)
 return 0 if result['status']=='all_events_committed' else 1


if __name__=='__main__':raise SystemExit(main())
