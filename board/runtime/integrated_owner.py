"""One SDK owner: initialize configurable resident graphs, then run only native control.

No runtime CPU book, normalization, reference inference or tensor expansion.
After a finite run allocations remain held until G's verified software reboot.
"""
from pathlib import Path
from datetime import timedelta
import argparse,ctypes,hashlib,importlib,io,json,os,re,sys,time,threading,traceback
import numpy as np
from startup_input import prepare_input, prepare_mapping

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(str(tmp),str(p))

def run(d,session_id,first,last,audit_inputs,slots=32,max_age_us=2000,h50_records=50,trace_capacity=32768,runtime_ms=180000):
 if slots not in (2,4,8,16,32) or not 0<=max_age_us<=(2**63-1)//125 or not 0<=h50_records<=65535:raise ValueError('Queue configuration range')
 cfg=json.loads((d/'config.json').read_text());refs=[]
 state=dict(status='initializing',pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
  session=session_id,first=first,last=last,queue_slots=slots,max_age_us=max_age_us,H50_records=h50_records,input_bytes=4800,FPGA_book=True,FPGA_preprocessing=True,FPGA_encoding=True,
  FPGA_output_read_and_argmax=True,CPU_reference=False,per_window_input_DMA=False,per_window_output_DMA=False)
 save(d/'progress.json',state)
 try:
  os.sched_setaffinity(0,{1})
  for name,digest in cfg['files'].items():
   if sha(d/name)!=digest:raise ValueError('Bundle changed '+name)
  lib=ctypes.CDLL(str(d/'libintegrated.so'));refs.append(lib)
  lib.dq_open.argtypes=[ctypes.c_uint32];lib.dq_reg.argtypes=[ctypes.c_uint32];lib.dq_reg.restype=ctypes.c_uint32
  lib.sf_read.argtypes=[ctypes.c_uint32];lib.sf_read.restype=ctypes.c_uint32;lib.sf_write.argtypes=[ctypes.c_uint32,ctypes.c_uint32]
  lib.dq_input.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
  lib.gi_book_snapshot.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint64)]
  lib.gi_snapshot.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint64)]
  sys.path.insert(0,'/usr/lib/aarch64-linux-gnu')
  xir,xrt,host,zg=[importlib.import_module(n) for n in ('xir','xrt','host_backend','zg330backend')]
  device=xrt.Device.Open('axi://zg330aiu');refs.append(device);region=device.defaultMemRegion()
  if 'plddr' not in region.typeKey().lower():raise ValueError('PL DDR region required')
  orders=region.malloc(8*1024*1024,False,4096)
  padding_bytes=(-(int(orders.begin.addr())+8*1024*1024))&262143
  padding=region.malloc(padding_bytes,False,4096) if padding_bytes else None
  if padding is not None:refs.append(padding)
  inputs=[region.malloc(8192,False,4096) for _ in range(slots)];output=region.malloc(4096,False,4096)
  scratch=device.getMemRegion('udma').malloc(16384,False,4096);refs.extend([orders,*inputs,output,scratch])
  addresses=[int(c.begin.addr()) for c in [orders,*inputs,output]]
  save(d/'allocation.json',dict(addresses=addresses,padding_bytes=padding_bytes,padding_address=int(padding.begin.addr()) if padding else None))
  if addresses[1]&0x3ffff or any(address!=addresses[1]+i*8192 for i,address in enumerate(addresses[1:-1])):raise ValueError('SDK inputs must form contiguous8KiB chunks from256KiB aligned base')
  extents=[(addresses[0],8*1024*1024)]+[(a,8192) for a in addresses[1:-1]]+[(addresses[-1],4096)]
  for i,(base,size) in enumerate(extents):
   if base&4095 or base+size>2**32:raise ValueError('SDK alignment/range')
   for other,length in extents[:i]:
    if max(base,other)<min(base+size,other+length):raise ValueError('SDK extents overlap')
  if lib.dq_open(int(scratch.begin.addr())):raise RuntimeError('Native MMIO/UDMA open')
  if [lib.sf_read(v) for v in (0xd0000,0xd0004,0xd0008,0xd2000)]!=[0x31424746,1,0,0x34514746]:raise ValueError('Clean integrated firmware required')
  first_input=np.fromfile(d/'golden_input.i8',dtype='u1');expected=np.fromfile(d/'golden_logits.f32',dtype='<f4')
  if first_input.size!=4800 or expected.size!=3:raise ValueError('Known input ABI')
  recipes=[];bindings=[];known=[]
  # One graph owns all weights, workspaces and instruction state. DDR queue
  # capacity is independent: clone only the input entry of its MMU table.
  for slot in range(1):
   state.update(phase='qualifying_resident_graphs',initializing_slot=slot,known=known);save(d/'progress.json',state)
   net=xir.Network.CreateFromJsonFile(str(d/'model.json'));net.loadParamsFromFile(str(d/'model.raw'));view=net.view(1)
   typ=view.inputs()[0].tensorType();input_id=int(view.inputs()[0].v_id)
   if list(typ.shape)!=[1,3,1,100,16] or int(typ.bytes())!=4800:raise ValueError('Actual INT8 storage ABI')
   s=xrt.Session.Create([zg.ZG330Backend,host.HostBackend],view,[device,xrt.HostDevice.Default()])
   backend=zg.ZG330Backend(s.backends[0]);backend.userConnectNetwork(inputs[slot],input_id);backend.userConnectNetwork(output,cfg['output_id']);s.apply()
   forwards=s.getForwards();ops=[(op,b) for op,b,*_ in forwards if 'HardOp' in op.typeKey() and 'ZG330' in b.typeKey()]
   if len(ops)!=1:raise ValueError('One resident NPU block required')
   forward_ops=[dict(op=op.typeKey(),backend=b.typeKey()) for op,b,*_ in forwards]
   if len(forward_ops)!=2 or any(not (('HardOp' in f['op'] and 'ZG330' in f['backend']) or ('Output' in f['op'] and 'Host' in f['backend'])) for f in forward_ops):raise ValueError('Unexpected runtime Host compute/adapter: '+str(forward_ops))
   hop=backend.forward_info.hardop_map[int(ops[0][0].op_id)]
   if int(hop.sync_idx[1])!=30:raise ValueError('Actual completion layer count')
   info=backend.forward_info
   for ident,address,size in ((input_id,addresses[slot+1],4800),(cfg['output_id'],addresses[-1],12)):
    chunk=info.memchunk_map[ident];value=info.value_map[ident]
    if not chunk.user_used or chunk.staged_chunk or int(value.phy_addr)!=address or int(value.byte_size)<size or value.is_host or value.is_ocm:raise ValueError('Direct SDK binding changed')
   tensor=xrt.Tensor(typ,inputs[slot],0)
   def callback(session,op,b,tensors):
    if 'HardOp' in op.typeKey():
     for t in tensors:
      t.setReady(False)
      if not t.waitForReady(timedelta(seconds=5),timedelta(0)):raise TimeoutError('NPU initialization')
   s.setPostCallBack(callback)
   refs.extend([net,view,s,backend,tensor,callback])
   state['golden_input_transfer']=prepare_input(inputs[slot],first_input,slot,d,'known-model-startup')
   save(d/'startup_inputs.json',dict(golden=state['golden_input_transfer']))
   slot_values=[]
   for iteration in range(11):
    # SDK initialization switches 32 distinct MMU tables/instruction buffers.
    # Apply the same explicit idle-cache invalidation as the qualified native
    # dispatch before SDK forward. This runs only during initialization.
    lib.sf_write(0x4084c,15);lib.sf_write(0x4084c,0)
    lib.sf_write(0x42000,0);lib.sf_write(0x42028,1);lib.sf_write(0x44028,0)
    before=lib.dq_reg(0x40c64);tensor.setReady(True);outs=s.forward([tensor]);outs[0].setReady(False)
    if not outs[0].waitForReady(timedelta(seconds=5),timedelta(0)):raise TimeoutError('Known final output')
    stream=io.BytesIO();outs[0].dump(stream,'SFB');value=np.frombuffer(stream.getvalue(),dtype='<f4').copy()
    after=lib.dq_reg(0x40c64)
    if ((after-before)&0xffffffff)!=30 or value.tobytes()!=expected.tobytes():
     failure=dict(slot=slot,iteration=iteration,count_before=before,count_after=after,completion_delta=(after-before)&0xffffffff,actual_logits=value.tolist(),expected_logits=expected.tolist(),actual_bits=value.view('<u4').tolist(),expected_bits=expected.view('<u4').tolist())
     try:
      check=np.empty(4800,dtype='u1');inputs[slot].read(check,0,len(check))
      failure.update(input_readback_mismatches=int(np.count_nonzero(check!=first_input)),input_expected_sha256=hashlib.sha256(first_input.tobytes()).hexdigest(),input_actual_sha256=hashlib.sha256(check.tobytes()).hexdigest())
     except Exception as diagnostic_error:failure['input_readback_error']=str(diagnostic_error)
     state['known_failure']=failure;save(d/'known_failure.json',failure)
     raise ValueError('Known NPU output or completion count differs: '+json.dumps(failure))
    slot_values.append(value.tolist())
   recipe=[lib.dq_reg(o) for o in (0x42018,0x4201c,0x42020)]+[int(hop.sync_idx[0]),30]+[lib.dq_reg(o) for o in (0x40840,0x40844,0x40848)]
   recipes.extend(recipe);known.append(dict(slot=slot,invocations=11,byte_equal=True,recipe=recipe,explicit_idle_MMU_VPE_flush=True))
   bindings.append(dict(slot=slot,input_address=addresses[slot+1],output_address=addresses[-1],instruction_address=int(hop.instr_phy_addr),instruction_bytes=int(hop.instr_size),layers=30,MMU_mode=recipe[5],MMU_table=recipe[6],MMU_config=recipe[7],forwards=forward_ops))
  input_description=next(v for op in json.loads((d/'model.json').read_text())['ops'] for v in op.get('outputs',[]) if int(v['v_id'])==input_id)
  match=re.fullmatch(r'@chunk\((\d+)@0\)',input_description['mtype'])
  if not match:raise ValueError('Input must have its own chunk at offset zero')
  input_chunk_id=int(match.group(1))
  # Python does not expose the table allocation. Snapshot its programmed map
  # page read-only for this frozen, single resident graph (<1024 chunk IDs).
  # This is a page snapshot, not a claim that the logical table is 4096 bytes.
  table_bytes=padded_bytes=4096;original_table_address=recipe[6]
  if recipe[5]!=1 or original_table_address&63:raise ValueError('Unsupported resident MMU table mode/alignment')
  lib.gi_read_bytes.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
  table=np.zeros(padded_bytes//4,dtype='<u4')
  if lib.gi_read_bytes(original_table_address,table.ctypes.data,padded_bytes):raise RuntimeError('Resident MMU map page snapshot')
  if input_chunk_id>=len(table) or int(table[input_chunk_id])!=addresses[1]>>8:raise ValueError('SDK input MMU entry is not the observed physical address/256')
  original_table=table.copy();base_recipe=recipe.copy();base_binding=bindings[0].copy()
  recipes=[];bindings=[];mapping_records=[]
  for slot in range(slots):
   mapped=original_table.copy();mapped[input_chunk_id]=addresses[slot+1]>>8
   chunk=region.malloc(padded_bytes,False,256);refs.append(chunk);table_address=int(chunk.begin.addr())
   if table_address&255 or table_address+padded_bytes>2**32:raise ValueError('Cloned MMU table alignment/range')
   # Initialization only: keep SDK-managed translation-table writes and reads
   # on the same SDK transfer path. Never mix a native upload with a following
   # SDK read of a newly allocated table. Runtime tensors remain zero-copy.
   transfer=prepare_mapping(chunk,mapped.view('u1'),slot,d,table_address)
   changed=np.flatnonzero(mapped!=original_table).tolist()
   if changed!=([input_chunk_id] if slot else []):raise ValueError('Clone changed non-input MMU mappings')
   r=base_recipe.copy();r[6]=table_address;recipes.extend(r)
   b=base_binding.copy();b.update(slot=slot,input_address=addresses[slot+1],MMU_table=table_address,resident_graph=0);bindings.append(b)
   mapping_records.append(dict(slot=slot,table_address=table_address,bytes=padded_bytes,input_chunk=input_chunk_id,changed_entries=changed,sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),readback_equal=True,initialization_upload='SDK MemChunk.write',transfer=transfer))
  state.update(resident_graphs=1,independent_DDR_slots=slots,input_mapping_clones=mapping_records)
  state['keep_resident_vpe']=False
  save(d/'resident_mappings.json',dict(status='passed',resident_graphs=1,original_table_address=original_table_address,table_snapshot_bytes=table_bytes,source_table_logical_size_not_exposed_by_Python=True,input_chunk_id=input_chunk_id,mappings=mapping_records,non_input_entries_unchanged=True))
  # Two different inputs detect a stale VPE stream when switching resident
  # instruction addresses. A repeated identical golden input cannot do this.
  probe_inputs=np.fromfile(d/'dual_slot_input.i8',dtype='u1').reshape(2,4800)
  probe_logits=np.fromfile(d/'dual_slot_logits.f32',dtype='<f4').reshape(2,3)
  lib.gi_dispatch.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
  lib.dq_output.argtypes=[ctypes.c_uint32,ctypes.c_void_p]
  probe_recipes=np.array(recipes,dtype='<u4').reshape(slots,8)
  input_transfers=[]
  for slot in range(slots):
   input_transfers.append(prepare_input(inputs[slot],probe_inputs[slot%2],slot,d,'alternating-native-slot-probe'))
  save(d/'startup_inputs.json',dict(golden=state['golden_input_transfer'],slots=input_transfers,all_verified_before_NPU=True))
  lib.gi_read_bytes.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
  instruction_blobs=[]
  for binding in bindings:
   blob=np.empty(binding['instruction_bytes'],dtype='u1')
   if lib.gi_read_bytes(binding['instruction_address'],blob.ctypes.data,len(blob)):raise RuntimeError('Startup instruction read')
   instruction_blobs.append(blob.tobytes())
  identical_instructions=all(v==instruction_blobs[0] for v in instruction_blobs)
  # This is one actual graph, not instruction aliasing of independent sessions.
  # All non-input MMU entries, weights and workspaces are exactly shared.
  shared_instructions=True
  state['instruction_bytes_identical']=identical_instructions
  state['shared_instruction_stream']=shared_instructions
  state['instruction_hashes']=[hashlib.sha256(v).hexdigest() for v in instruction_blobs]
  dual_probe=[]
  for slot in list(range(slots))+list(reversed(range(slots))):
   counts=np.zeros(2,dtype='<u4');value=np.zeros(3,dtype='<f4')
   if lib.gi_dispatch(probe_recipes[slot].ctypes.data,counts.ctypes.data) or lib.dq_output(addresses[-1],value.ctypes.data):raise RuntimeError('Native dual-slot dispatch')
   if int((int(counts[1])-int(counts[0]))&0xffffffff)!=30:raise ValueError('Startup native layer count')
   dual_probe.append(dict(slot=slot,counts=counts.tolist(),logits=value.tolist(),byte_equal=value.tobytes()==probe_logits[slot%2].tobytes()))
   state['dual_slot_qualification']=dual_probe
   save(d/'dual_slot_qualification.json',dict(status='running',observations=dual_probe))
   if not dual_probe[-1]['byte_equal']:
    failure=dict(slot=slot,recipe=probe_recipes[slot].tolist(),counts=counts.tolist(),actual=value.tolist(),expected=probe_logits[slot%2].tolist())
    check=np.empty(4800,dtype='u1');inputs[slot].read(check,0,len(check));bad=np.flatnonzero(check!=probe_inputs[slot%2])
    failure.update(input_readback_mismatches=int(len(bad)),mismatch_offsets=bad[:64].tolist(),expected_bytes=probe_inputs[slot%2][bad[:64]].tolist(),actual_bytes=check[bad[:64]].tolist(),input_verified_before_NPU=input_transfers[slot]['mismatched_bytes']==0)
    save(d/'native_slot_failure.json',failure);probe_inputs[slot%2].tofile(d/'native_input_expected.bin');check.tofile(d/'native_input_actual.bin')
    state['native_qualification_failure']=failure
    raise ValueError('Native slot output differs from same-graph Host: '+json.dumps(failure))
  save(d/'dual_slot_qualification.json',dict(status='passed',observations=dual_probe,per_window_input_DMA=False,scope='Startup qualification only'))
  state.update(phase='initializing_order_arena',known=known,dual_slot_qualification=dual_probe,bindings=bindings,addresses=addresses,device_versions=dict(device.version()))
  save(d/'progress.json',state)
  zero=np.zeros(8192,dtype='u1')
  for off in range(0,8*1024*1024,8192):
   if lib.dq_input(addresses[0]+off,zero.ctypes.data,8192):raise RuntimeError('Order arena initialization')
  check=np.empty(8192,dtype='u1')
  for off in (0,8*1024*1024-8192):
   orders.read(check,off,len(check))
   if np.any(check):raise ValueError('Order arena endpoint readback')
  from configure_phy import configure
  phy=configure(d)
  if phy['link_mbps']!=1000 or not phy['full_duplex']:raise RuntimeError('PL cable needs1Gb/full link')
  def wr(off,value):lib.sf_write(0xd0000+off,int(value)&0xffffffff)
  def wr64(off,value):wr(off,value);wr(off+4,value>>32)
  value=ctypes.c_uint64()
  def book_status(index):
   if lib.gi_book_snapshot(index,ctypes.byref(value)):raise TimeoutError('Book status')
   return value.value
  def predict_status(index):
   if lib.gi_snapshot(0xd2000,index,ctypes.byref(value)):raise TimeoutError('Prediction status')
   return value.value
  wr64(0x10,session_id);wr(0x18,addresses[0]);wr(0x1c,0xc0a87e01);wr64(0x20,first);wr64(0x28,last);wr(0x30,2);wr(0x80,0x53544152)
  end=time.monotonic()+5
  while book_status(8)&6!=6:
   if book_status(8)&1 or time.monotonic()>end:raise RuntimeError('Book initialization')
  wr(0x2800,addresses[1])
  for off,val in [(0x2020,addresses[-1]),(0x2040,max_age_us*125),(0x2044,(max_age_us*125)>>32),(0x2048,slots),(0x204c,h50_records),(0x2024,0x41524d34)]:wr(off,val)
  end=time.monotonic()+5
  while predict_status(0)&6!=6:
   if predict_status(0)&1 or time.monotonic()>end:raise RuntimeError('Prediction initialization')
  trace_dtype=np.dtype([(n,'<u8') for n in ['window','ordinal','slot','generation','request_ns','npu_ns','board_ticks','service_ticks']]+[('logits','<f4',(3,)),('class_index','<u4'),('count_before','<u4'),('count_after','<u4')])
  if not 1<=trace_capacity<=500000 or not 1000<=runtime_ms<=3600000:raise ValueError('Finite trace/runtime bounds')
  trace=np.zeros(trace_capacity,dtype=trace_dtype);recipe_array=probe_recipes.flatten().copy();stop=ctypes.c_uint32();stats=(ctypes.c_uint64*4)();execution={}
  audit_dtype=np.dtype([('window','<u8'),('ordinal','<u8'),('generation','<u8'),('slot','<u8'),('input','u1',(4800,)),('logits','<f4',(3,)),('class_index','<u4')])
  audit=np.zeros(audit_inputs,dtype=audit_dtype)
  lib.gi_audit_set.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
  audit_bases=np.array(addresses[1:-1],dtype='<u4');lib.gi_audit_set(audit_bases.ctypes.data,slots,audit.ctypes.data,len(audit));refs.extend([audit,audit_bases])
  monitor=np.zeros(12,dtype='<u8');monitor_copy=np.zeros(12,dtype='<u8')
  lib.gi_monitor_set.argtypes=[ctypes.c_void_p];lib.gi_monitor_set(monitor.ctypes.data)
  lib.gi_monitor_copy.argtypes=[ctypes.c_void_p]
  lib.gi_trace_count.argtypes=[ctypes.POINTER(ctypes.c_uint64)];lib.gi_trace_count.restype=ctypes.c_uint64
  refs.extend([monitor,monitor_copy]);state['monitor_mode']='5Hz native sole-GP-reader; display only'
  lib.gi_run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint32]
  def worker():execution['code']=lib.gi_run(recipe_array.ctypes.data,trace.ctypes.data,len(trace),first,last,ctypes.byref(stop),runtime_ms,stats,slots)
  thread=threading.Thread(target=worker);thread.start();refs.extend([trace,recipe_array,stop,stats,thread])
  state.update(status='ready',phy=phy,measured_trace_capacity=len(trace),native_core=2,control_core=1)
  save(d/'ready.json',state);finish_sent=False;next_display=0
  while thread.is_alive():
   if (d/'finish.request').exists() and not finish_sent:
    wr(0x84,0x454e4421);finish_sent=True;stop.value=1
   if time.monotonic()>=next_display:
    next_display=time.monotonic()+.2
    n=int(lib.gi_trace_count(stats))
    def display_trace(t):return dict(window=int(t['window']),ordinal=int(t['ordinal']),slot=int(t['slot']),generation=int(t['generation']),npu_ns=int(t['npu_ns']),reason=int(t['class_index'])>>24)
    latest=[display_trace(t) for t in trace[max(0,n-8):n]]
    initial=[display_trace(t) for t in trace[:min(n,8)]]
    if lib.gi_monitor_copy(monitor_copy.ctypes.data)==0:
     values=monitor_copy.tolist()
     names=['records','opportunities','admitted','retired','snapshot_full','expired','occupied','queue_peak']
     save(d/'ui_telemetry.json',dict(schema='finedge-native-monitor-v2',sample_board_ns=values[1],sample_interval_ms=200,metrics=dict(zip(names,values[2:10])),initial_native=initial,recent_native=latest,trace_count=n,GP_reader='native worker only'))
   # The native worker is the sole GP reader until it has proven all rows,
   # encoding and admitted terminals drained. No concurrent Python MMIO reads.
   time.sleep(.005)
  thread.join();trace[:stats[2]].tofile(d/'traces.bin')
  audit_count=lib.gi_audit_count();audit[:audit_count].tofile(d/'audited_inputs.bin')
  metrics={str(i):predict_status(i) for i in range(41)}
  book_metrics={hex(i):book_status(i) for i in range(0,0x158,8)}
  if execution['code'] or metrics['0']&1 or metrics['3']!=metrics['5'] or metrics['2']!=metrics['3']+metrics['6']+metrics['7']+metrics['8'] or metrics['9']!=0 or metrics['16']!=0 or metrics['23']!=0 or metrics['13']!=40*metrics['12']:raise RuntimeError('Finite run failed: '+str(execution))
  state.update(transport_version='window_stream_v1',whole_window_completion=True,full_snapshot_RAM_copies=0,status='finished_parked',native=execution,statistics=list(stats),traces=int(stats[2]),metrics=metrics,book_metrics=book_metrics,finish_sent=finish_sent,trace_bytes=int(stats[2])*trace_dtype.itemsize,audited_inputs=audit_count,audit_enabled=bool(audit_inputs))
  save(d/'result.json',state)
 except BaseException as exc:
  state.update(status='failed_parked',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());save(d/'result.json',state)
 while True:time.sleep(30)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--session',type=int,required=True);p.add_argument('--first',type=int,required=True);p.add_argument('--last',type=int,required=True);p.add_argument('--audit-inputs',type=int,default=0,choices=[0,2,32]);p.add_argument('--slots',type=int,default=32,choices=[2,4,8,16,32]);p.add_argument('--max-age-us',type=int,default=2000);p.add_argument('--h50-records',type=int,default=50);p.add_argument('--trace-capacity',type=int,default=32768);p.add_argument('--runtime-ms',type=int,default=180000);a=p.parse_args();run(a.directory.resolve(),a.session,a.first,a.last,a.audit_inputs,a.slots,a.max_age_us,a.h50_records,a.trace_capacity,a.runtime_ms)
