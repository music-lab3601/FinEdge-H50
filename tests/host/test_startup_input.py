from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np
U=Path(__file__).resolve().parents[1]
from board.runtime.startup_input import prepare_input, prepare_mapping

class MemoryChunk:
    def __init__(self,corrupt=False):self.memory=np.zeros(8192,dtype='u1');self.corrupt=corrupt;self.events=[]
    def write(self,offset,data,count):
        self.events.append('write');self.memory[offset:offset+count]=data
        if self.corrupt:self.memory[71]^=1
    def read(self,data,offset,count):self.events.append('read');data[:]=self.memory[offset:offset+count]

class StartupInputTest(unittest.TestCase):
    def test_one_byte_fault_is_detected_before_any_NPU_launch_and_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            chunk=MemoryChunk(True);payload=(np.arange(4800)%256).astype('u1')
            with self.assertRaisesRegex(ValueError,'before NPU'):prepare_input(chunk,payload,3,d,'probe',max_attempts=1)
            failure=json.loads((Path(d)/'startup_input_failure.json').read_text())
            self.assertEqual(failure['offsets'],[71]);self.assertEqual(failure['mismatched_bytes'],1)
            self.assertEqual(chunk.events,['write','read','read']);self.assertNotEqual(failure['expected_sha256'],failure['actual_sha256'])
            self.assertEqual((Path(d)/'startup_input_expected.bin').read_bytes(),payload.tobytes())
    def test_distinct_slot_patterns_are_written_and_checked_without_retries(self):
        with tempfile.TemporaryDirectory() as d:
            for slot in range(32):
                chunk=MemoryChunk();payload=np.full(4800,slot,dtype='u1');r=prepare_input(chunk,payload,slot,d,'probe')
                self.assertEqual(r['mismatched_bytes'],0);self.assertEqual(r['expected_sha256'],r['actual_sha256']);self.assertEqual(chunk.events,['write','read','read'])
    def test_transient_fault_retransfers_and_requires_two_clean_reads(self):
        class Transient(MemoryChunk):
            def write(self,offset,data,count):
                super().write(offset,data,count)
                if self.events.count('write')==1:self.memory[71]^=1
        with tempfile.TemporaryDirectory() as d:
            chunk=Transient();r=prepare_input(chunk,np.zeros(4800,dtype='u1'),19,d,'probe')
            self.assertEqual(len(r['attempts']),2);self.assertEqual(r['recovered_transfer_attempts'],1)
            self.assertEqual(r['consecutive_equal_reads'],2);self.assertTrue(r['verified_before_NPU'])
            self.assertTrue((Path(d)/'startup_input_attempts.jsonl').exists())
    def test_persistent_fault_stops_after_three_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            chunk=MemoryChunk(True)
            with self.assertRaises(ValueError):prepare_input(chunk,np.zeros(4800,dtype='u1'),3,d,'probe')
            self.assertEqual(chunk.events.count('write'),3)

class MappingTransferTest(unittest.TestCase):
    def test_transient_mapping_fault_is_retransferred_and_evidence_kept(self):
        class Transient(MemoryChunk):
            def write(self,offset,data,count):
                super().write(offset,data,count)
                if self.events.count('write')==1:self.memory[3076]^=32
        with tempfile.TemporaryDirectory() as d:
            chunk=Transient();payload=(np.arange(4096)%256).astype('u1')
            r=prepare_mapping(chunk,payload,16,d,9159424)
            self.assertEqual(r['recovered_transfer_attempts'],1);self.assertEqual(r['consecutive_equal_reads'],2)
            self.assertEqual(r['table_address'],9159424);self.assertTrue(r['verified_before_NPU'])
            failed=json.loads((Path(d)/'mmu_attempts.jsonl').read_text())
            self.assertEqual(failed['offsets'],[3076]);self.assertTrue((Path(d)/'mmu_slot_16_attempt_1_first.bin').exists())
    def test_first_good_read_cannot_mask_second_read_corruption(self):
        class UnstableRead(MemoryChunk):
            def read(self,data,offset,count):
                super().read(data,offset,count)
                if self.events.count('read')==2:data[3082]^=1
        with tempfile.TemporaryDirectory() as d:
            chunk=UnstableRead();r=prepare_mapping(chunk,np.zeros(4096,dtype='u1'),16,d,9159424)
            self.assertEqual(r['attempts'][0]['first_read_mismatches'],0)
            self.assertEqual(r['attempts'][0]['second_read_mismatches'],1)
            self.assertEqual(len(r['attempts']),2)
    def test_persistent_mapping_fault_stops_and_saves_exact_payload(self):
        with tempfile.TemporaryDirectory() as d:
            chunk=MemoryChunk(True);payload=np.zeros(4096,dtype='u1')
            with self.assertRaisesRegex(ValueError,'before NPU'):prepare_mapping(chunk,payload,16,d,9159424)
            self.assertEqual(chunk.events.count('write'),3)
            failure=json.loads((Path(d)/'mmu_failure.json').read_text())
            self.assertEqual(failure['consecutive_equal_reads'],0)
            self.assertEqual((Path(d)/'mmu_expected.bin').read_bytes(),payload.tobytes())

if __name__=='__main__':unittest.main()
