"""Independent bounded E candidate. Transport injection is the only device seam.

The CLI exposes only an offline transport substitute. No owner lifecycle,
reset, retry, free or termination operation is provided by this module.
"""
from contextlib import contextmanager
from time import perf_counter_ns
import numpy as np
from experiments.reference.runtime.udw_lab import UdwLab, LabFault, Slot, digest
from experiments.reference.runtime.udw_runtime import CapacityBudget
from experiments.reference.runtime.udw_e_frontend import CandidateFrontend
from experiments.reference.runtime.ddr_transport_contract import Descriptor, OUTPUT_STRIDE, unpack_record, unpack_output


class Timeline:
    def __init__(self):self.spans=[]

    @contextmanager
    def measure(self,name):
        begin=perf_counter_ns()
        try:yield
        finally:self.spans.append(dict(name=name,begin_ns=begin,end_ns=perf_counter_ns()))

    def summary(self,begin,end):
        last=begin;totals={}
        for s in self.spans:
            if not begin<=s['begin_ns']<=s['end_ns']<=end or s['begin_ns']<last:
                raise ValueError('Overlapping or out-of-bound timing spans')
            last=s['end_ns'];totals[s['name']]=totals.get(s['name'],0)+s['end_ns']-s['begin_ns']
        total=end-begin;measured=sum(totals.values())
        return dict(total_ns=total,exclusive_ns=totals,unattributed_ns=total-measured,
                    account_balanced=(measured<=total))


def require_context(context,expected,arm=None):
    """Validate caller's new-session observations BEFORE invoking an owner factory.

    Not a physical-stop proof. Board adapters must obtain trusted fresh readings;
    this round's executable supplies explicit offline fixtures only.
    """
    if context['boot_id']==context['previous_boot_id'] or not context['boot_id']:
        raise ValueError('New cold boot identity required')
    if context['physical_stop_confirmed'] is not True:raise ValueError('Physical stop confirmation required')
    for k in ('BOOT_sha256','owner_sha256','ZG_json_sha256','ZG_raw_sha256','SDK_version'):
        if context[k]!=expected[k]:raise ValueError('Startup provenance mismatch: '+k)
    if context['offline_only'] is not True:
        from experiments.reference.runtime.udw_e_release import BoardPermit
        permit=context.get('board_permit')
        if not isinstance(permit,BoardPermit):raise ValueError('Reviewed board permit required')
        permit.validate(arm)


class CandidateSession(UdwLab):
    """Reuse accepted health/sequence/wait logic, replace only audit-heavy submit.

    No expected tensor construction in submit/predict. Actual images and hashes
    are retained; full references are compared outside the timed phase.
    """
    def __init__(self,transport,base,size,*,run_id,generation=1,journal=lambda r:None,
                 timeout=5,quota_bytes=512*1024):
        self.budget=CapacityBudget(size,quota_bytes)
        self.used=0;self.current=None;self.current_image=None;self.expected_regions=[]
        self.phase='setup';self.timeline=None
        self.last_attempt=None
        super().__init__(transport,base,size,run_id=run_id,generation=generation,
                         journal=journal,timeout=timeout)

    def guard(self,point):
        s=super().guard(point)
        required=(4,8,12,0x54,0x58,0x60,0x64,0x68,0x6c,0x70,0x8c,0xc0)
        if any(k not in s or type(s[k]) is not int or not 0<=s[k]<2**32 for k in required):
            self.fail(point+': incomplete/invalid status')
        return s

    @contextmanager
    def timing(self,name):
        if self.timeline is None:yield
        else:
            with self.timeline.measure(name):yield

    def capacity(self,size):return self.budget.decision(self.used,size)

    def reserve_fresh(self,size):
        if self.failed:raise LabFault('Experiment already quarantined')
        if not self.capacity(size)['fits']:raise LabFault('Capacity stop before reserve')
        offset=self.t.reserve(size)
        if type(offset) is not int or offset!=self.used or offset%64 or offset+size>self.size:
            self.fail('Nonmonotonic/overlapping/out-of-range allocation')
        self.used+=size
        return offset

    def setup_scratch(self):
        if self.failed:raise LabFault('Experiment already quarantined')
        if self.used or self.phase!='setup':raise LabFault('Scratch setup cannot be repeated')
        try:
            offset=self.reserve_fresh(4096);image=bytes(range(256))*16
            self.t.prepare(offset,image)
            if self.t.read(offset,len(image))!=image:self.fail('SDK scratch self-check')
            self.expected_regions.append((offset,image))
        except Exception as e:self.fail('Scratch setup failed: '+str(e))

    def submit(self,records,infos,*,continuation=False):
        if self.failed:raise LabFault('Experiment already quarantined')
        try:return self._submit_candidate(records,infos,continuation)
        except Exception as e:
            if not self.failed:self.fail('Submit stopped: '+str(e))
            raise

    def _submit_candidate(self,records,infos,continuation):
        if not records or len(records)!=len(infos):raise ValueError('Record/info count mismatch')
        selected=[i for i in infos if i['selected']]
        for record,info in zip(records,infos,strict=True):
            words,identity,emit=unpack_record(record)
            if (identity!=info['identity'] or emit!=info['selected']
                or words!=tuple(int(v) for v in info['normalized'].view('<u4'))):
                raise ValueError('WDM metadata/normalized payload mismatch')
        data=b''.join(records);capacity=max(64,len(selected)*OUTPUT_STRIDE)
        size=len(data)+capacity+256
        # Capacity is checked BEFORE even a REARM, not just before reservation.
        if not self.capacity(size)['fits']:raise LabFault('Capacity stop before command/reserve')
        with self.timing('descriptor_prestate'):
            s=self.guard('before_next_request')
            if s[8]==2:self.issue(2,0)
            elif s[8]!=0:self.fail('Unexpected prestate',s)
        with self.timing('reserve_prepare_upload'):
            offset=self.reserve_fresh(size)
            slot=Slot(offset,size,offset+64,offset+len(data)+192,len(data),capacity)
            image=self.GUARD+data+self.GUARD*2+bytes([self.POISON])*capacity+self.GUARD
            self.slots.append(slot)
            self.last_attempt=dict(request_id=self.request_id+1,offset=offset,size=size,
                input_sha256=digest(data),command_may_have_arrived=False,completed=False)
            self.t.prepare(offset,image)
            if self.t.read(offset,size)!=image:self.fail('Current prepare image mismatch')
            self.guard('after_prepare')
        request=self.request_id+1
        d=Descriptor(self.base+slot.input_offset,len(data),self.base+slot.output_offset,
                     capacity,len(records),len(selected),self.run_id,request,self.generation)
        registers=d.shadow_registers(address_limit_exclusive=self.base+self.size)
        with self.timing('descriptor_command_wait'):
            self.t.registers(registers);shadow=self.guard('after_descriptor')
            if any(shadow.get(k)!=v for k,v in registers.items()):self.fail('Shadow descriptor mismatch')
            self.last_attempt['command_may_have_arrived']=True # conservative before issue
            terminal=self.issue(4 if continuation else 1,2)
            self.last_attempt['terminal_observed_ns']=perf_counter_ns()
            counts={0x40:len(records),0x44:len(selected),0x48:len(selected)*OUTPUT_STRIDE,
                    0x4c:len(records)*4,0x50:len(selected)*252,0x58:1}
            if any(terminal.get(k)!=v for k,v in counts.items()):self.fail('Terminal count/capability mismatch')
        with self.timing('current_readback_checks'):
            self.guard('before_output_read');actual=self.t.read(offset,size);self.guard('after_output_read')
            relative=slot.output_offset-offset;valid=len(selected)*OUTPUT_STRIDE
            if (len(actual)!=size or actual[:relative]!=image[:relative]
                or actual[relative+valid:]!=image[relative+valid:]):
                self.fail('Current input/guard/poison image mismatch')
            tensors=[]
            for i,info in enumerate(selected):
                tensor,first,last=unpack_output(actual[relative+i*OUTPUT_STRIDE:relative+(i+1)*OUTPUT_STRIDE],
                    run_id=self.run_id,request_id=request,generation=self.generation,window_index=i)
                if first!=info['first'] or last!=info['last']:self.fail('Output window endpoint mismatch')
                if not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all():self.fail('Nonfinite output tensor')
                tensors.append(tensor)
            self.guard('after_output_verify')
            self.current=slot;self.current_image=actual
            self.expected_regions.append((offset,actual));self.request_id=request
            self.last_attempt['completed']=True
        return tensors

    def check_current(self):
        if self.current is not None:
            if self.t.read(self.current.offset,self.current.length)!=self.current_image:
                self.fail('Current region changed around NPU call')

    def predict(self,tensor,window_id):
        if self.failed:raise LabFault('Experiment already quarantined')
        try:
            if len(tensor)!=16000 or not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all():
                raise ValueError('Invalid NPU input')
            self.guard('before_npu');self.check_current()
            result=self.t.call('predict',hex=tensor.hex(),window_id=window_id)
            self.guard('after_npu');self.check_current()
            if result['window_id']!=window_id:raise ValueError('NPU window identity mismatch')
            values=np.asarray(result['logits'])
            if values.shape!=(3,) or not np.isfinite(values).all():raise ValueError('Invalid finite logits')
            return result
        except Exception as e:
            if not self.failed:self.fail('NPU stopped: '+str(e))
            raise

    def audit(self):
        # Explicit setup/post boundary only; never called by hot submit/predict.
        if self.phase=='hot':raise LabFault('Full history audit inside hot path prohibited')
        try:super().verify_retained()
        except Exception as e:
            if not self.failed:self.fail('Boundary audit failed: '+str(e))
            raise


class CandidateRunner:
    """Public S/H/C path shared by dry-run evidence and future board adapter."""
    def __init__(self,normalization,arm,observer=lambda stage,report:None):
        self.arm=arm;self.frontend=CandidateFrontend(normalization,arm)
        self.failed=False;self.session=None;self.report={};self.next_id=0;self.consumed=False
        self.observer=observer

    def fail(self,error):
        self.failed=True
        self.report.update(status='failed',error=f'{type(error).__name__}: {error}',
            quarantined=True,owner_retained=True,strict_visibility_proven=False)
        if self.session is not None and not self.session.failed:
            try:self.session.fail(str(error))
            except LabFault:pass

    def run(self,batches,references,*,context,expected_context,session_factory=None,cpu_factory=None):
        if self.failed:raise LabFault('Runner quarantined; no resume')
        if self.consumed:raise LabFault('One-shot runner already consumed; new cold session required')
        self.consumed=True
        clock=perf_counter_ns;begin=clock();hot=Timeline();cpu=None
        r=self.report=dict(status='running',arm=self.arm,board_accessed=False,
            backend_execution='offline_transport_substitute' if self.arm!='C' else 'CPU',
            predictions=[],batches=[],record_times=[],records_submitted=0,selected_submitted=0,
            strict_visibility_proven=False,owner_retained=True,default_runtime_changed=False,
            performance_acceptance=False,hot_spans=[])
        def predict(tensor):
            self.next_id+=1
            if self.session is not None:return self.session.predict(tensor,self.next_id)
            result=cpu.predict(self.next_id,np.frombuffer(tensor,dtype='<f4').reshape(1,40,100))
            if result['window_id']!=self.next_id or np.asarray(result['logits']).shape!=(3,) or not np.isfinite(result['logits']).all():
                raise ValueError('CPU output identity/finite logits mismatch')
            return result
        def check_prediction(tensor,result):
            target=references['logits'][digest(tensor)]['cpu' if self.arm=='C' else 'npu']
            if not np.allclose(result['logits'],target,rtol=1e-4 if self.arm=='C' else 1e-3,
                               atol=1e-5 if self.arm=='C' else 1e-4):
                raise ValueError('Outside-hot numerical reference mismatch')
        try:
            require_context(context,expected_context,self.arm)
            real_board=context['offline_only'] is not True
            substitute=context.get('transport_substitute') is True
            r['board_accessed']=real_board and not substitute
            if real_board:r['backend_execution']='orchestration_substitute' if substitute else 'ZG_socket_actual_board'
            self.observer('context_verified',r)
            if self.arm!='C':
                self.session=session_factory()
                self.session.setup_scratch()
                minimal=references['minimal']
                got=self.session.submit(minimal['wdm'],minimal['infos'])
                if got!=[minimal['tensor']]:raise ValueError('Minimal 100-to-1 mismatch')
                self.observer('minimal_window_verified',r)
            else:cpu=cpu_factory()
            # Loading/session creation/allocation belong to setup, once only.
            for _ in range(10):predict(references['fixed'][0])
            for tensor in references['fixed']:
                result=predict(tensor);check_prediction(tensor,result)
            result=predict(references['minimal']['tensor']);check_prediction(references['minimal']['tensor'],result)
            if self.session:self.session.audit();self.session.phase='hot';self.session.timeline=hot
            self.observer('before_hot',r)
            hot_begin=clock();ready_ns=hot_begin # finite burst: all raw records ready now
            r['hot_begin_ns']=hot_begin
            for batch_index,batch in enumerate(batches,1):
                if len(batch)!=64:raise ValueError('Initial E candidate requires 64-record batches')
                infos=[];payloads=[]
                for record,identity in batch:
                    with hot.measure('validation_normalization_ring'):
                        info=self.frontend.push(record,identity)
                    with hot.measure('WDM_pack' if self.arm=='H' else 'software_window_pack'):
                        payload=self.frontend.pack(info)
                    r['record_times'].append(dict(identity=list(info['identity']),raw_ready_ns=ready_ns,
                        process_begin_ns=hot.spans[-2]['begin_ns'],normalization_ring_end_ns=hot.spans[-2]['end_ns'],
                        pack_end_ns=hot.spans[-1]['end_ns'],selected=info['selected']))
                    infos.append(info);payloads.append(payload)
                selected=[info for info in infos if info['selected']]
                size=len(batch)*256+max(64,len(selected)*OUTPUT_STRIDE)+256
                # Apply H's same budget to S/C to enforce identical bounded input.
                used=self.session.used if self.session and self.arm=='H' else r.get('logical_used',46080)
                decision=CapacityBudget(2*1024*1024,512*1024).decision(used,size)
                r['batches'].append(dict(index=batch_index,records=len(batch),selected=len(selected),
                    capacity_fits=decision['fits'],submitted=False,completed=False,
                    capacity=decision,release_ns=clock()))
                if not decision['fits']:
                    r['stopped_before_batch']=batch_index;break
                if self.arm=='H':
                    with hot.measure('submit_validation_and_wrapper'):
                        # Inner transport spans are recorded in a distinct ledger
                        # and not added to this parent interval.
                        self.session.timeline=None
                        bridge=Timeline();self.session.timeline=bridge
                        try:tensors=self.session.submit(payloads,infos,continuation=batch_index>1)
                        finally:self.session.timeline=None
                    r['batches'][-1]['bridge_subspans']=bridge.spans
                    r['batches'][-1]['request_observation']=dict(self.session.last_attempt)
                else:
                    tensors=[p for p,i in zip(payloads,infos,strict=True) if i['selected']]
                r['batches'][-1].update(submitted=True,completed=True)
                r['logical_used']=used+size;r['records_submitted']+=len(batch)
                for info,tensor in zip(selected,tensors,strict=True):
                    with hot.measure('predict_with_health_and_completion'):
                        result=predict(tensor)
                    inference_span=hot.spans[-1]
                    with hot.measure('result_identity_and_hash'):
                        if self.failed:raise LabFault('Quarantined')
                        row=dict(first=list(info['first']),last=list(info['last']),tensor_sha256=digest(tensor),
                            result=result,class_index=int(np.argmax(result['logits'])),
                            predict_begin_ns=inference_span['begin_ns'],predict_end_ns=inference_span['end_ns'],
                            raw_ready_ns=ready_ns,result_available_ns=clock())
                        row['age_ns']=row['result_available_ns']-ready_ns
                        r['predictions'].append(row)
                    r['selected_submitted']+=1
            hot_end=clock();r['hot_end_ns']=hot_end;r['hot_spans']=hot.spans
            r['timing']=hot.summary(hot_begin,hot_end)
            if self.session:self.session.phase='post';self.session.timeline=None;self.session.audit()
            self.observer('hot_complete_audit_done',r)
            # All expensive references/Host-model comparisons outside hot path.
            if (r['records_submitted']!=references['expected_records']
                or [tuple(row['last']) for row in r['predictions']]!=references['expected_ids']):
                raise ValueError('Missing/extra records or selected windows in completed run')
            for row in r['predictions']:
                ref=references['windows'][tuple(row['last'])]
                if row['first']!=list(ref['first']) or row['tensor_sha256']!=digest(ref['tensor']):
                    raise ValueError('Post-audit software reference tensor/identity mismatch')
                check_prediction(ref['tensor'],row['result'])
            for tensor in references['fixed']:
                result=predict(tensor);check_prediction(tensor,result)
            if self.session:self.session.audit()
            status=('orchestration_substitute_passed' if substitute else 'bounded_board_candidate_verified') if real_board else 'local_candidate_passed'
            r.update(status=status,quarantined=False,
                setup_ns=hot_begin-begin,post_ns=clock()-hot_end)
        except Exception as e:
            self.fail(e)
        finally:
            # No close/free/terminate on any path, including exception/EOF.
            end=clock();r['round_total_ns']=end-begin;r['prediction_calls']=self.next_id
            r['hot_spans']=hot.spans
            if self.session is not None:
                r['session_observation']=dict(used_bytes=self.session.used,failed=self.session.failed,
                    started_or_possibly_started=self.session.started,last_attempt=self.session.last_attempt,
                    request_id=self.session.request_id,run_id=self.session.run_id,generation=self.session.generation,
                    slots=[s.__dict__ for s in self.session.slots])
            if 'timing' in r:
                r['accounting_residual_ns']=r['round_total_ns']-r['setup_ns']-r['timing']['total_ns']-r['post_ns'] if r['status'] in ('local_candidate_passed','bounded_board_candidate_verified','orchestration_substitute_passed') else None
                r['round_amortized_ns_per_window']=r['round_total_ns']/max(1,r['selected_submitted'])
            r['H_CPU_tensor_ring_allocated']=self.arm=='H' and self.frontend.ring is not None
        return r
