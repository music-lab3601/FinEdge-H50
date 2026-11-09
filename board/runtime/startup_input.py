"""SDK-only startup input transfer, before live FPGA/zero-copy inference."""
from pathlib import Path
import hashlib,json
import numpy as np

def prepare_input(chunk,payload,slot,directory,purpose,max_attempts=3):
    expected=np.ascontiguousarray(payload,dtype=np.uint8)
    if expected.ndim!=1 or expected.size!=4800:raise ValueError('Startup INT8 input must be4800 bytes')
    if not 1<=max_attempts<=3:raise ValueError('Startup transfer attempts must be1..3')
    d=Path(directory);attempts=[]
    for attempt in range(1,max_attempts+1):
        # Only owned startup test memory is re-transferred. No producer is
        # enabled, no NPU is running, and numerical qualification stays strict.
        chunk.write(0,expected,len(expected))
        actual=np.empty_like(expected);confirm=np.empty_like(expected)
        chunk.read(actual,0,len(actual));chunk.read(confirm,0,len(confirm))
        bad=np.flatnonzero((actual!=expected)|(confirm!=expected))
        observation=dict(attempt=attempt,first_read_mismatches=int(np.count_nonzero(actual!=expected)),second_read_mismatches=int(np.count_nonzero(confirm!=expected)),
                         first_read_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),second_read_sha256=hashlib.sha256(confirm.tobytes()).hexdigest())
        attempts.append(observation)
        result=dict(slot=int(slot),purpose=purpose,bytes=len(expected),write_path='SDK MemChunk.write/read',
                    mismatched_bytes=int(len(bad)),expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    actual_sha256=observation['first_read_sha256'],verified_before_NPU=not len(bad),
                    attempts=attempts,recovered_transfer_attempts=attempt-1,consecutive_equal_reads=2 if not len(bad) else 0)
        if not len(bad):return result
        result.update(offsets=bad[:64].tolist(),expected=expected[bad[:64]].tolist(),actual=actual[bad[:64]].tolist(),confirmation=confirm[bad[:64]].tolist())
        with (d/'startup_input_attempts.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(result)+'\n')
        # Preserve failed reads even when a later verified transfer succeeds.
        prefix='startup_slot_'+str(slot)+'_attempt_'+str(attempt)
        actual.tofile(d/(prefix+'_first.bin'));confirm.tofile(d/(prefix+'_second.bin'))
    (d/'startup_input_failure.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    expected.tofile(d/'startup_input_expected.bin');actual.tofile(d/'startup_input_actual.bin')
    raise ValueError('Startup input differs before NPU: '+json.dumps(result))


def prepare_mapping(chunk,payload,slot,directory,table_address,max_attempts=3):
    """Upload an owned, inactive MMU clone; require two exact SDK readbacks.

    Called before any clone is installed or live FPGA input is enabled. An
    incomplete mapping is never submitted to the NPU. This is not runtime
    repair and does not allow numerical qualification to be bypassed.
    """
    expected=np.ascontiguousarray(payload,dtype=np.uint8)
    if expected.ndim!=1 or expected.size!=4096:raise ValueError('MMU page snapshot must be4096 bytes')
    if not 1<=max_attempts<=3:raise ValueError('MMU transfer attempts must be1..3')
    d=Path(directory);attempts=[]
    for attempt in range(1,max_attempts+1):
        chunk.write(0,expected,len(expected))
        actual=np.empty_like(expected);confirm=np.empty_like(expected)
        chunk.read(actual,0,len(actual));chunk.read(confirm,0,len(confirm))
        bad=np.flatnonzero((actual!=expected)|(confirm!=expected))
        observation=dict(attempt=attempt,first_read_mismatches=int(np.count_nonzero(actual!=expected)),
                         second_read_mismatches=int(np.count_nonzero(confirm!=expected)),
                         first_read_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                         second_read_sha256=hashlib.sha256(confirm.tobytes()).hexdigest())
        attempts.append(observation)
        result=dict(slot=int(slot),table_address=int(table_address),bytes=len(expected),
                    write_path='SDK MemChunk.write/read',mismatched_bytes=int(len(bad)),
                    expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    actual_sha256=observation['first_read_sha256'],attempts=attempts,
                    recovered_transfer_attempts=attempt-1,consecutive_equal_reads=2 if not len(bad) else 0,
                    verified_before_NPU=not len(bad))
        if not len(bad):return result
        result.update(offsets=bad[:64].tolist(),expected=expected[bad[:64]].tolist(),
                      actual=actual[bad[:64]].tolist(),confirmation=confirm[bad[:64]].tolist())
        with (d/'mmu_attempts.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(result)+'\n')
        prefix='mmu_slot_'+str(slot)+'_attempt_'+str(attempt)
        actual.tofile(d/(prefix+'_first.bin'));confirm.tofile(d/(prefix+'_second.bin'))
    (d/'mmu_failure.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    expected.tofile(d/'mmu_expected.bin');actual.tofile(d/'mmu_actual.bin')
    raise ValueError('Cloned MMU table differs before NPU: '+json.dumps(result))
