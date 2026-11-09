"""Explicit shared-input protocol semantics over the bounded PST1 codec."""
from __future__ import annotations
import math
from contextlib import nullcontext

from experiments.reference.runtime.ps_stream_protocol import ClientCodec, ServerCodec, Kind, ProtocolError
from experiments.reference.runtime.ps_window_runtime import selected_endpoints

SCHEMA='finedge-shared-input-plddr-v1'


class SharedServer:
    def __init__(self,controller,*,token,max_windows,identity,observe=lambda row:None,batch_scope=nullcontext):
        self.controller=controller
        self.codec=ServerCodec(token,max_windows)
        self.identity=dict(identity,schema=SCHEMA)
        self.observe=observe
        self.batch_scope=batch_scope

    def serve(self,read,write):
        while self.codec.state!='FINISHED':
            request=self.codec.receive_request(read)
            windows=[]
            if request.kind==Kind.BATCH:
                expected,_=selected_endpoints(request.payload,self.controller.history)
                if len(expected)>request.window_budget:raise ProtocolError('Shared window budget too small')
                # Keep the PC->board batch of up to 32 records. The fixed
                # NPU input slot is consumed serially inside this one request.
                with self.batch_scope():
                    pending=b''
                    for offset in range(0,len(request.payload),256):
                        pending+=request.payload[offset:offset+256]
                        ready,_=selected_endpoints(pending,self.controller.history)
                        if ready or offset+256==len(request.payload):
                            result=self.controller.submit(pending);pending=b''
                            for row in result['windows']:
                                row=dict(row,start=row['metadata']['first'],end=row['metadata']['last'])
                                windows.append(row)
                                self.observe(row)
            elif request.kind==Kind.FINISH and self.controller.owner.state!='FREE':
                raise ProtocolError('NPU/FPGA ownership still outstanding')
            self.codec.respond(write,windows=windows,data=dict(identity=self.identity,controller=self.controller.summary()))
        return self.controller.summary()


class SharedClient:
    def __init__(self,read,write,*,token,max_windows,expected_identity):
        self.read,self.write=read,write
        self.codec=ClientCodec()
        self.token,self.max_windows=token,max_windows
        self.expected_identity=dict(expected_identity,schema=SCHEMA)

    def _receive(self):
        response=self.codec.receive_response(self.read)
        identity=response['data'].get('identity',{})
        if any(identity.get(k)!=v for k,v in self.expected_identity.items()):
            self.codec.state='FAILED';raise ProtocolError('Shared session identity differs')
        return response

    def hello(self):
        self.codec.send_hello(self.write,self.token,self.max_windows)
        self.token=None
        return self._receive()

    def submit(self,payload,expected=None):
        expected=[] if expected is None else [expected] if isinstance(expected,dict) else expected
        if not isinstance(expected,list):raise ValueError('Explicit bounded expected window list required')
        self.codec.send_batch(self.write,payload,len(expected))
        response=self._receive()
        rows=response['windows']
        try:
            if len(rows)!=len(expected):raise ValueError('Wrong shared window count')
            for index,(row,reference) in enumerate(zip(rows,expected)):
                metadata=row['metadata']
                if (row['window_id']!=self.codec.total_windows-len(rows)+index+1 or row['start']!=list(reference['first'])
                        or row['end']!=list(reference['last']) or row.get('input_consumed') is not True):
                    raise ValueError('Shared window/consumption identity mismatch')
                for k in ('crc32','sample_sha256'):
                    if k in reference and metadata.get(k)!=reference[k]:raise ValueError('Shared sampled input differs')
                values=row['logits']
                if len(values)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
                    raise ValueError('Invalid NPU logits')
                if type(row['class_index']) is not int or row['class_index']!=max(range(3),key=lambda i:values[i]):
                    raise ValueError('Class/logits mismatch')
                if type(row.get('whole_predict_ns')) is not int or row['whole_predict_ns']<0:
                    raise ValueError('Invalid formal NPU call timing')
            return response
        except BaseException:
            self.codec.state='FAILED';raise

    def finish(self):
        self.codec.send_finish(self.write)
        return self._receive()
