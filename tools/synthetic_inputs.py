"""Deterministic public MBO fixtures and a UDP/CPU functional replay."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from contracts.market.finedge_contract import RAWMBO, decode_event, event_packet
from deployment.icraft.native.codec3 import pack
from host.telemetry.telemetry import ROW
from shared.io import atomic_json, contained_file, sha
from shared.paths import ROOT, ConfigurationError
from software.data.databento_mbo_v2 import Adapter, midnight_ns
from software.data.itch_book import Book, snapshot

SCHEMA = 'finedge-public-synthetic-inputs-v1'
DAY = '2025-12-15'
ROWS = 3050
START_ORDINAL = 21
FILES = dict(raw_source='market/raw56.bin', reference='display/rows.bin',
             duration_data='ranges.json', golden_input='startup/golden_input.i8',
             golden_logits='startup/golden_logits.f32', dual_input='startup/dual_slot_input.i8',
             dual_logits='startup/dual_slot_logits.f32')
STARTUP = dict(golden_input='golden_input.i8', golden_logits='golden_logits.f32',
               dual_input='dual_slot_input.i8', dual_logits='dual_slot_logits.f32')


def model_identity():
    return {name: sha(ROOT / path) for name, path in dict(
        model='models/npu/model.json', parameters='models/npu/model.raw',
        normalization='models/normalization.json', scales='models/npu/scales.f32').items()}


def startup_bytes():
    contract = json.loads((ROOT / 'contracts/examples/synthetic_startup.json').read_text('utf-8'))
    if contract['model_identity'] != model_identity():
        raise ConfigurationError('Synthetic startup reference belongs to a different frozen model')
    first = pack(np.zeros((1, 40, 100), dtype=np.int8))
    codes = ((np.arange(4000, dtype=np.int32).reshape(1, 40, 100) % 127) - 63).astype(np.int8)
    second = pack(codes)
    logits = bytes.fromhex(contract['Host_logits_f32le_hex'])
    result = dict(golden_input=first, golden_logits=logits[:12],
                  dual_input=first + second, dual_logits=logits)
    if {name: hashlib.sha256(blob).hexdigest() for name, blob in result.items()} != contract['artifact_sha256']:
        raise ConfigurationError('Synthetic startup bytes differ from the SDK-verified reference')
    return result


def raw_events():
    origin = midnight_ns(DAY) + (9 * 3600 + 30 * 60) * 1_000_000_000

    def record(ordinal, stamp, action, order=0, side='N', price=0, quantity=0):
        return RAWMBO.pack(14, 160, 2, 1, stamp, order, price * 100000,
                           quantity, 128, 0, ord(action), ord(side), stamp, 0, ordinal)

    yield record(1, origin - 200_000_000, 'R')
    for index in range(20):
        side = 'B' if index % 2 == 0 else 'A'
        price = 1000000 - (index // 2) * 100 if side == 'B' else 1000200 + (index // 2) * 100
        yield record(index + 2, origin - (19 - index) * 10_000_000,
                     'A', index + 1, side, price, 100 + index * 11)
    for index in range(1, ROWS):
        order = (index - 1) % 20 + 1
        side = 'B' if order % 2 else 'A'
        price = 1000000 - ((order - 1) // 2) * 100 if side == 'B' else 1000200 + ((order - 1) // 2) * 100
        phase = (index // 20) % 80
        drift = (phase if phase < 40 else 79 - phase) - 20
        yield record(index + 21, origin + index * 100_000_000,
                     'M', order, side, price + drift, 100 + (index * 17 + order * 11) % 400)


def apply_raw(adapter, blob):
    values = RAWMBO.unpack(blob)
    if values[:4] != (14, 160, 2, 1) or values[8] != 128:
        raise ValueError('Synthetic raw identity or event boundary differs')
    action = chr(values[10])
    if action == 'R':
        adapter.book = Book()
        return None
    adapter.mutate(SimpleNamespace(action=action, order_id=values[5], price=values[6],
                                   size=values[7], side=chr(values[11])))
    flags, row = snapshot(adapter.book, regular=True, tradable=True)
    return None if flags else np.asarray(row, dtype='<u8')


def reference_rows(events):
    adapter = Adapter(DAY, {1}, lambda *args: None)
    rows = []
    for ordinal, blob in enumerate(events, 1):
        values = apply_raw(adapter, blob)
        if values is not None:
            rows.append((1, len(rows) + 1, ordinal, RAWMBO.unpack(blob)[4], values))
    adapter.book.validate()
    return np.array(rows, dtype=ROW)


def dataset_bytes():
    events = list(raw_events())
    rows = reference_rows(events)
    if len(rows) != ROWS:
        raise RuntimeError('Synthetic book row count differs')
    return b''.join(events), rows


def ranges():
    result = dict(schema='finedge-public-synthetic-ranges-v1', data_kind='synthetic', ranges={})
    for minutes in (1, 2, 5):
        count = minutes * 600
        result['ranges'][str(minutes)] = dict(id=f'synthetic_{minutes}m', minutes=minutes,
            duration_ns=minutes * 60_000_000_000, first_raw_ordinal=START_ORDINAL,
            last_raw_ordinal=20 + count, measured_events=count, bootstrap_events=20,
            event_span_ns=(count - 1) * 100_000_000, eligible_rows=count, opportunities=count - 99)
    return result


def generate(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw, rows = dataset_bytes()
    blobs = dict(raw_source=raw, reference=rows.tobytes(), **startup_bytes())
    range_bytes = (json.dumps(ranges(), indent=2) + '\n').encode('utf-8')
    blobs['duration_data'] = range_bytes
    files = {}
    for name, blob in blobs.items():
        path = directory / FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        files[name] = dict(path=FILES[name], bytes=len(blob), sha256=sha(path))
    manifest = dict(schema=SCHEMA, generator='tools.synthetic_inputs.v1', data_kind='synthetic',
                    symbol='SYNTH', date=DAY, calendar_scope='Artificial timestamps on a fixed EST calendar',
                    eligibility='Synthetic regular/tradable gate with ten uncrossed levels',
                    events=len(raw) // 56, rows=len(rows), records_per_second=10,
                    model_identity=model_identity(), files=files, market_accuracy_evaluated=False,
                    NPU_reference_backend='Icraft 3.36.1 Host', board_validation='deferred')
    atomic_json(directory / 'manifest.json', manifest)
    return verify(directory / 'manifest.json')


def load_profile(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text('utf-8'))
    if value.get('schema') != SCHEMA or value.get('data_kind') != 'synthetic' or value.get('model_identity') != model_identity():
        raise ConfigurationError('Unrecognized synthetic profile or frozen model identity')
    if set(value['files']) != set(FILES):
        raise ConfigurationError('Synthetic profile file list differs')
    resolved = {}
    for name, row in value['files'].items():
        if row['path'] != FILES[name]:
            raise ConfigurationError('Synthetic profile relative path differs')
        target = contained_file(path.parent, row['path'])
        if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
            raise ConfigurationError('Synthetic input missing or changed: ' + name)
        resolved[name] = target
    for name, blob in startup_bytes().items():
        if resolved[name].read_bytes() != blob:
            raise ConfigurationError('Synthetic SDK startup reference differs')
    return value, resolved


def verify(path):
    value, files = load_profile(path)
    raw, rows = dataset_bytes()
    if files['raw_source'].read_bytes() != raw or files['reference'].read_bytes() != rows.tobytes():
        raise ConfigurationError('Synthetic MBO/book reconstruction differs from generator')
    if json.loads(files['duration_data'].read_text('utf-8')) != ranges():
        raise ConfigurationError('Synthetic duration ranges differ')
    return dict(status='passed', data_kind='synthetic', events=len(raw) // 56, rows=len(rows),
                windows_5m=2901, files=len(files), board_access=False)


def cpu_replay(profile, directory, records=160, speed=10):
    if type(records) is not int or not 100 <= records <= ROWS or type(speed) is not int or not 1 <= speed <= 1000:
        raise ValueError('Expected 100..3050 records and replay speed 1..1000')
    verify(profile)
    _, files = load_profile(profile)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from shared.model_paths import MANIFEST_SHA, PACKAGE
    from software.runtime.cpu import Predictor, preprocess_raw
    from software.runtime.stream import InputBuilder, Record
    normalization = json.loads((ROOT / 'models/normalization.json').read_text('utf-8'))
    predictor = Predictor(PACKAGE, MANIFEST_SHA)
    builder = InputBuilder(normalization, packing='single_copy')
    adapter = Adapter(DAY, {1}, lambda *args: None)
    raw = files['raw_source'].read_bytes()[:(20 + records) * RAWMBO.size]
    reference = np.fromfile(files['reference'], dtype=ROW)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(('127.0.0.1', 0))
    receiver.settimeout(5)
    address = receiver.getsockname()
    stop = threading.Event()
    faults, publications = [], []

    def send():
        try:
            anchor = time.perf_counter_ns() + 50_000_000
            origin = RAWMBO.unpack(raw[:56])[4]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as source:
                for index in range(len(raw) // 56):
                    blob = raw[index * 56:(index + 1) * 56]
                    due = anchor + (RAWMBO.unpack(blob)[4] - origin) // speed
                    if stop.wait(max(0, due - time.perf_counter_ns()) / 1e9):
                        return
                    submitted = time.perf_counter_ns()
                    source.sendto(event_packet(blob, 1, index + 1, index + 1), address)
                    publications.append(dict(ordinal=index + 1, due_ns=due, submitted_ns=submitted,
                                             send_done_ns=time.perf_counter_ns()))
        except Exception as exc:
            faults.append(str(exc))

    sender = threading.Thread(target=send, name='public-synthetic-UDP-source')
    sender.start()
    rows, outputs = [], []
    try:
        for index in range(len(raw) // 56):
            packet = decode_event(receiver.recvfrom(2048)[0])
            if packet['packet_seq'] != index + 1 or packet['first_raw_ordinal'] != index + 1 or packet['session_id'] != 1 or packet['records'] != 1:
                raise ValueError('Synthetic UDP sequence, identity or record count differs')
            values = apply_raw(adapter, packet['raw'])
            if values is None:
                continue
            row = len(rows)
            np.testing.assert_array_equal(values, reference[row]['values'])
            rows.append(values)
            metadata = dict(date=DAY, symbol='SYNTH', snippet_id=1, research_segment_id=1,
                native_segment_id=1, research_reject_mask=0, bid_levels=10, ask_levels=10,
                source_record_row_0based=row, sample_row_0based=row,
                source_message_position_1based=index + 1,
                timestamp_ns_since_exchange_midnight=RAWMBO.unpack(packet['raw'])[4] - midnight_ns(DAY))
            window = builder.push(Record(values, metadata))
            if window is not None:
                np.testing.assert_array_equal(window.tensor, preprocess_raw(np.stack(rows[-100:])[None], normalization))
                prediction = predictor.predict_normalized(window.tensor)
                outputs.append(dict(window=row - 98, ordinal=index + 1,
                    tensor_sha256=hashlib.sha256(window.tensor.tobytes()).hexdigest(),
                    logits=prediction['logits'][0].tolist(), class_index=int(prediction['class_indices'][0])))
        adapter.book.validate()
    finally:
        stop.set()
        sender.join(timeout=5)
        receiver.close()
    if faults or len(rows) != records or len(outputs) != records - 99:
        raise RuntimeError('Incomplete synthetic UDP/CPU replay: ' + str(faults))
    atomic_json(directory / 'publications.json', publications)
    atomic_json(directory / 'predictions.json', outputs)
    result = dict(status='passed', data_kind='synthetic', backend='CPU', network='loopback UDP',
                  events=len(raw) // 56, records=len(rows), windows=len(outputs), speed=speed,
                  model_identity=model_identity(), input_manifest_sha256=sha(profile),
                  CPU_manifest_sha256=MANIFEST_SHA, CPU_model_sha256=sha(PACKAGE / 'model_cpu.ts'),
                  market_accuracy_evaluated=False, board_access=False)
    atomic_json(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    generate_args = sub.add_parser('generate')
    generate_args.add_argument('--directory', type=Path, required=True)
    verify_args = sub.add_parser('verify')
    verify_args.add_argument('--profile', type=Path, required=True)
    cpu_args = sub.add_parser('cpu')
    cpu_args.add_argument('--profile', type=Path, required=True)
    cpu_args.add_argument('--directory', type=Path, required=True)
    cpu_args.add_argument('--records', type=int, default=160)
    cpu_args.add_argument('--speed', type=int, default=10)
    bundle_args = sub.add_parser('bundle')
    bundle_args.add_argument('--profile', type=Path, required=True)
    bundle_args.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'generate':
        result = generate(args.directory)
    elif args.action == 'verify':
        result = verify(args.profile)
    elif args.action == 'cpu':
        result = cpu_replay(args.profile, args.directory, args.records, args.speed)
    else:
        from shared.config import board_config
        from tools.board_bundle import prepare_integrated_bundle
        verify(args.profile)
        directory = args.directory.resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        config = prepare_integrated_bundle(directory, board_config(profile=args.profile))
        result = dict(status='prepared', files=len(config['files']), data_kind='synthetic',
                      NPU_reference_backend='Icraft Host', board_validation='deferred', board_access=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
