"""Generated MBO -> UDP -> visible book -> frozen preprocessing -> CPU model."""
import hashlib
import json
import socket
import threading
import time

import numpy as np

from contracts.market.finedge_contract import RAWMBO, decode_event, event_packet
from shared.model_paths import MANIFEST_SHA, PACKAGE
from shared.paths import ROOT, work_path
from software.data.itch_book import Book, snapshot
from software.runtime.cpu import Predictor, preprocess_raw
from software.runtime.stream import InputBuilder, Record


def run(records=160):
    if type(records) is not int or not 100 <= records <= 10000:
        raise ValueError('Smoke records must be 100..10000')
    normalization = json.loads((ROOT / 'models/normalization.json').read_text('utf-8'))
    predictor = Predictor(PACKAGE, MANIFEST_SHA)
    builder = InputBuilder(normalization, packing='single_copy')
    book = Book()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0))
    receiver.settimeout(5)
    address = receiver.getsockname()
    faults = []
    total = records + 20

    def send():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as source:
                for i in range(total):
                    side = ord('B') if i % 2 == 0 else ord('A')
                    price = (1000000 - (i % 20) * 100) if side == ord('B') else (1000200 + (i % 20) * 100)
                    raw = RAWMBO.pack(14, 160, 1, 1, i * 1000000, i + 1,
                                      price * 100000, 100 + i % 80, 0, 0, ord('A'), side, 0, 0, i + 1)
                    source.sendto(event_packet(raw, 1, i + 1, i + 1), address)
                    time.sleep(.001)
        except BaseException as exc:
            faults.append(str(exc))

    sender = threading.Thread(target=send, name='synthetic-market-source')
    sender.start()
    rows, predictions = [], []
    started = time.perf_counter_ns()
    try:
        for i in range(total):
            packet = decode_event(receiver.recvfrom(2048)[0])
            if packet['packet_seq'] != i + 1 or packet['session_id'] != 1:
                raise ValueError('Synthetic UDP identity mismatch or loss')
            raw = RAWMBO.unpack(packet['raw'])
            book.add(raw[5], 'B' if raw[11] == ord('B') else 'S', raw[6] // 100000, raw[7])
            flags, values = snapshot(book, True, True)
            if flags or len(rows) >= records:
                continue
            index = len(rows)
            values = np.asarray(values, dtype='<u8')
            rows.append(values)
            meta = dict(date='synthetic', symbol='SYNTH', snippet_id=1, research_segment_id=1,
                        native_segment_id=1, research_reject_mask=0, bid_levels=10, ask_levels=10,
                        source_record_row_0based=index, sample_row_0based=index,
                        source_message_position_1based=i + 1, timestamp_ns_since_exchange_midnight=raw[4])
            window = builder.push(Record(values, meta))
            if window is not None:
                expected = preprocess_raw(np.stack(rows[-100:])[None], normalization)
                np.testing.assert_array_equal(window.tensor, expected)
                result = predictor.predict_normalized(window.tensor)
                predictions.append(dict(window=index - 98, logits=result['logits'][0].tolist(),
                                        class_index=int(result['class_indices'][0]),
                                        tensor_sha256=hashlib.sha256(window.tensor.tobytes()).hexdigest()))
        book.validate()
    finally:
        receiver.close()
        sender.join(timeout=10)
    if faults or len(rows) != records or len(predictions) != records - 99:
        raise RuntimeError('Incomplete synthetic network loop. ' + str(faults))
    result = dict(status='passed', input='generated synthetic MBO over loopback UDP', backend='CPU',
                  events=total, eligible_records=len(rows), windows=len(predictions),
                  elapsed_ns=time.perf_counter_ns() - started, predictions=predictions, board_access=False)
    output = work_path('checks/cpu_network.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return {key: value for key, value in result.items() if key != 'predictions'}
