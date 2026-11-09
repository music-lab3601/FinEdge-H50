"""P1 registration and read-only screening, without labels, statistics or models."""
from shared.paths import resolve_path

import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/tcn_research_screening_p1_2026-10-14'
CFG = ROOT / 'configs/tcn_research_protocol_b_v1.json'
REGISTRY = OUT / 'source_registry.json'
DATA_ROOT = Path(str(resolve_path('data://research/FinEdge_LOB_Data'))).resolve()
UPSTREAM = ROOT / 'artifacts/tcn_data_v2_2026-10-09'
STATUS_DIR = ROOT / 'artifacts/tcn_databento_status_2026-10-10'
TRAIN = [f'2025-12-{d:02}' for d in range(8, 13)]
NEW = [f'2025-12-{d:02}' for d in range(15, 20)]
ROLES = {**dict.fromkeys(TRAIN, 'train'), '2025-12-15': 'tuning_validation',
         '2025-12-16': 'tuning_validation', '2025-12-17': 'comparison_validation',
         '2025-12-18': 'final_test', '2025-12-19': 'final_test'}
NS = 1_000_000_000
DAY_NS = 86_400 * NS
EPOCH_DATE = date(1970, 1, 1)
PROFILE = 'target-date-fixed-EST-1.0'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = bounded_path(path, (ROOT,), 'All outputs must stay in workspace')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def small_binding(path, expected=None):
    path = bounded_path(path, (ROOT, DATA_ROOT), 'Unapproved metadata root')
    need(path.is_file() and path.stat().st_size <= 2 * 1024**2, 'Metadata absent or too large')
    digest = sha(path)
    need(expected is None or digest == expected, 'Metadata identity changed: ' + str(path))
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def verify_binding(binding):
    return small_binding(binding['path'], binding['sha256'])


def bounded_path(path, roots, error):
    # Normalize '..' lexically before any resolve/stat on a rejected target.
    lexical = Path(os.path.abspath(os.fspath(path)))
    need(any(lexical.is_relative_to(root) for root in roots), error)
    resolved = lexical.resolve()
    need(any(resolved.is_relative_to(root) for root in roots), error)
    return resolved


def research_window_acceptance(summary):
    if not isinstance(summary, dict):
        return False
    count = summary.get('window_count')
    return summary.get('screening_passed') is True and type(count) is int and count > 0


def midnight_est_ns(day):
    need(day in ROLES, 'Fixed-EST profile refuses unregistered dates')
    # Exact integer conversion; UTC midnight + 05:00 is NY EST midnight.
    return ((date.fromisoformat(day) - EPOCH_DATE).days * 86_400 + 5 * 3600) * NS


def normalize_status_facts(facts, day, enum_maps, instrument_ids):
    result, previous_recv, previous_event = [], None, None
    origin = midnight_est_ns(day)
    required = {'record_ordinal', 'publisher_id', 'instrument_id', 'ts_event', 'ts_recv',
                'ts_index', 'action', 'action_enum', 'reason', 'reason_enum',
                'trading_event', 'trading_event_enum', 'is_trading', 'is_quoting',
                'is_short_sell_restricted'}
    for ordinal, fact in enumerate(facts, 1):
        need(isinstance(fact, dict) and required <= set(fact), 'Malformed STATUS fact')
        need(type(fact['record_ordinal']) is int and fact['record_ordinal'] == ordinal,
             'STATUS ordinal gap')
        need(type(fact['publisher_id']) is int and type(fact['instrument_id']) is int
             and fact['publisher_id'] == 2 and fact['instrument_id'] in instrument_ids,
             'STATUS publisher/instrument identity')
        event, recv = fact['ts_event'], fact['ts_recv']
        need(type(event) is int and type(recv) is int and origin <= event < origin + DAY_NS,
             'STATUS event is outside bound New York day')
        need(origin <= recv < origin + DAY_NS and fact['ts_index'] == recv,
             'STATUS receive/index identity')
        need(previous_recv is None or recv >= previous_recv, 'STATUS receive regression')
        need(previous_event is None or event >= previous_event, 'STATUS event regression')
        parsed = {}
        for field, class_name in [('action', 'StatusAction'), ('reason', 'StatusReason'),
                                  ('trading_event', 'TradingEvent')]:
            value = fact[field]
            need(type(value) is int, 'Enum must be an integer')
            need(type(fact[field + '_enum']) is str, 'Enum text must be a string')
            match = re.fullmatch(r'<' + class_name + r'\.([A-Z0-9_]+): ([0-9]+)>',
                                 fact[field + '_enum'])
            need(match is not None and int(match.group(2)) == value, 'Enum text/numeric mismatch')
            name = match.group(1)
            need(enum_maps[class_name].get(name) == value, 'Unknown or inconsistent locked enum')
            parsed[field] = name
        for field in ['is_trading', 'is_quoting', 'is_short_sell_restricted']:
            need(fact[field] in ('Y', 'N', '~'), 'Unknown tri-state')
        result.append({'day': day, 'ts_event': event - origin,
                       'action': parsed['action'], 'reason': parsed['reason'],
                       'is_trading': fact['is_trading'], 'is_quoting': fact['is_quoting'],
                       'ssr': fact['is_short_sell_restricted'], 'symbol': 'AAPL',
                       'source_ts_recv': recv, 'source_ts_event': event,
                       'source_record_ordinal': ordinal,
                       'source_publisher_id': fact['publisher_id'],
                       'source_instrument_id': fact['instrument_id'],
                       'source_trading_event': parsed['trading_event']})
        previous_recv, previous_event = recv, event
    need(bool(result), 'Missing STATUS facts')
    return result


def register():
    need(not CFG.exists() and not REGISTRY.exists(), 'Already registered; do not overwrite protocol')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('AGENTS.md', 'PROGRESS.md'):
        target = OUT / ('previous_' + name)
        if not target.exists():
            target.write_bytes((ROOT / name).read_bytes())
    protected = {}
    for folder in ['scripts', 'tests', 'configs', 'proposals', 'state', 'docs']:
        for p in (ROOT / folder).rglob('*'):
            if (p.is_file() and p.suffix.lower() in ('.py', '.json', '.txt', '.md')
                    and p.stat().st_size <= 2 * 1024**2
                    and p.name not in ('tcn_research_screening_core.py', 'tcn_research_screening_p1.py',
                                       'test_tcn_research_screening_core.py',
                                       'test_tcn_research_screening_p1.py')):
                protected[str(p.relative_to(ROOT))] = sha(p)
    write_json(OUT / 'protected_before.json', protected)
    completion = read_json(UPSTREAM / 'completion_acceptance.json')
    manifest = read_json(UPSTREAM / 'manifest.json')
    parent = read_json(ROOT / 'configs/tcn_data_protocol_v2.json')
    need(manifest['date_roles'] == parent['date_roles'] == ROLES, 'Date role drift')
    bindings = [small_binding(UPSTREAM / 'manifest.json', completion['outputs'][str(UPSTREAM / 'manifest.json')]),
                small_binding(ROOT / 'configs/tcn_data_protocol_v2.json', manifest['protocol_sha256']),
                small_binding(UPSTREAM / 'completion_acceptance.json'),
                small_binding(UPSTREAM / 'engineering_row_review.json', completion['outputs'][str(UPSTREAM / 'engineering_row_review.json')]),
                small_binding(ROOT / 'state/tcn_software_plan_v5.json')]
    engineering_scan = read_json(UPSTREAM / 'engineering_row_review.json')
    semantics = read_json(UPSTREAM / 'adapter_semantics_test_receipt.json')
    bindings.append(small_binding(UPSTREAM / 'adapter_semantics_test_receipt.json', completion['outputs'][str(UPSTREAM / 'adapter_semantics_test_receipt.json')]))
    for b in semantics['files']:
        bindings.append(small_binding(b['path'], b['sha256']))
    need(semantics['exit_code'] == 0 and semantics['test_count'] == 36, 'Engineering semantics not accepted')
    tech_receipt = read_json(STATUS_DIR / 'technical_execution_receipt.json')
    bindings.append(small_binding(STATUS_DIR / 'technical_execution_receipt.json'))
    need(tech_receipt['exit_code'] == 0, 'STATUS prior execution not accepted')
    for path, digest in tech_receipt['outputs_sha256'].items():
        bindings.append(small_binding(path, digest))
    download = read_json(STATUS_DIR / 'download_receipt.json')
    bindings.append(small_binding(STATUS_DIR / 'download_receipt.json', tech_receipt['download_receipt_sha256']))
    technical = read_json(STATUS_DIR / 'technical_review/status_technical_acceptance.json')
    need(technical['dependency']['databento-dbn'] == '0.71.0'
         and technical['source_identity_before_after_equal'], 'STATUS version/identity provenance')
    import databento_dbn as dbn
    need(importlib.metadata.version('databento-dbn') == '0.71.0', 'Locked enum dependency drift')
    enum_maps = {name: {k: int(getattr(getattr(dbn, name), k))
                        for k in dir(getattr(dbn, name)) if k.isupper()}
                 for name in ('StatusAction', 'StatusReason', 'TradingEvent')}
    sources = {}
    for day in ROLES:
        native = day in TRAIN
        item = copy.deepcopy(manifest['sources'][day] if native else manifest['diagnostic_new_sources'][day])
        p = Path(item['records_path']).resolve()
        need(p.is_relative_to(DATA_ROOT) and p.name == 'records.bin', 'Unapproved records path')
        need(p.parent.name.startswith('full_' if native else 'engineering_'), 'Failed/unknown source directory')
        need(p.stat().st_size == item['rows'] * 352, 'Record byte count mismatch')
        record = {k: item[k] for k in ('records_path', 'records_sha256', 'rows')}
        record.update({'day': day, 'role': ROLES[day], 'source_kind': 'native' if native else 'engineering',
                       'record_bytes': item['rows'] * 352, 'metadata_bindings': [],
                       'source_control_verified': native, 'coverage_unknown': not native})
        if native:
            pin = parent['old_source_versions'][day]
            need(item['records_sha256'] == pin['records_sha256'], 'Native source SHA differs from protocol')
            record['segments_path'] = item['segments_path']
            record['segments_sha256'] = item['segments_sha256']
            record['metadata_bindings'].append(small_binding(item['segments_path'], item['segments_sha256']))
            record['metadata_bindings'].append(small_binding(p.parent / 'report.json', pin['audit_report_sha256']))
            report = read_json(p.parent / 'report.json')
            need(report['symbol'] == 'AAPL' and report['trading_date'] == day and report['full_day']
                 and report['book_fatal_anomalies'] == 0 and report['source_sha256'] == pin['raw_sha256'],
                 'Native engineering report identity')
        else:
            local_report = UPSTREAM / ('adapter_' + day + '.json')
            record['metadata_bindings'].append(small_binding(local_report, completion['outputs'][str(local_report)]))
            report = read_json(local_report)
            need(read_json(p.parent / 'report.json') == report, 'Engineering source report changed')
            record['metadata_bindings'].append(small_binding(p.parent / 'report.json'))
            need(report['date'] == day and report['role'] == ROLES[day]
                 and Path(report['records_path']).resolve() == p
                 and report['records_sha256'] == item['records_sha256']
                 and report['rows'] == item['rows'] and report['records_bytes'] == item['rows'] * 352,
                 'Engineering report does not match manifest source identity')
            need(item['adapter_version'] == report['adapter_version'] == 'Databento-MBO-engineering-adapter-2.0',
                 'Engineering provenance version')
            need(report['metadata']['dataset'] == 'XNAS.ITCH' and report['metadata']['schema'] == 'mbo'
                 and report['metadata']['symbols'] == ['AAPL'], 'Engineering metadata identity')
            need(report['independent_row_scan_passed'] and engineering_scan['dates'][day]['record_sha256'] == item['records_sha256'],
                 'Original row-scan acceptance')
            counts = report['counts']
            need(counts.get('resets') == 1 and sum(v for k, v in counts.items() if k.startswith('native_pattern_')) == counts['complete_native_events'],
                 'Native group/initialization counts')
            need(sum(counts.get('native_pattern_' + k, 0) for k in ('A', 'C', 'CA', 'TFC', 'FC', 'M')) == item['rows'],
                 'Mutation group sampling counts')
            need(sum(report['provider_flag_counts'].values()) == counts['decoded_mbo_records']
                 and report['provider_flag_counts'].get('8') == 1
                 and set(report['provider_flag_counts']) <= {'0', '8', '128'}, 'Source flags/initial R evidence')
            need(sum(report['row_flag_counts'].values()) == item['rows'] and set(report['row_flag_counts']) <= {'81', '83'},
                 'Derived flags do not match accepted adapter profile')
            need(all(counts.get(k, 0) == 0 for k in ('event_timestamp_regressions', 'maybe_bad_book_records', 'bad_receive_timestamp_records')),
                 'Accepted adapter anomaly evidence')
            record['adapter_version'] = report['adapter_version']
            record['first_sample_event_unix_ns'] = report['first_sample_event_unix_ns']
            record['last_sample_event_unix_ns'] = report['last_sample_event_unix_ns']
            status_report = technical['days'][day]
            utc_origin = midnight_est_ns(day) - 5 * 3600 * NS
            need(report['metadata']['start'] == status_report['metadata']['start'] == utc_origin
                 and report['metadata']['end'] == status_report['metadata']['end'] == utc_origin + DAY_NS,
                 'MBO/STATUS UTC query coverage mismatch')
            mapping = report['metadata']['mappings']['AAPL']
            next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
            need(bool(mapping) and all(m['start_date'] == day and m['end_date'] == next_day for m in mapping)
                 and {int(m['symbol']) for m in mapping} == set(status_report['metadata']['instrument_ids']),
                 'MBO/STATUS daily instrument mapping mismatch')
            need(status_report['source_sha256'] == technical['files'][Path(status_report['source_path']).name]['sha256'],
                 'STATUS source hash chain')
            raw_binding = small_binding(status_report['source_path'], status_report['source_sha256'])
            record['metadata_bindings'].append(raw_binding)
            timeline = Path(status_report['timeline_path'])
            record['metadata_bindings'].append(small_binding(timeline, status_report['timeline_sha256']))
            facts = [json.loads(line) for line in timeline.read_text(encoding='utf-8').splitlines() if line.strip()]
            need(len(facts) == status_report['records'] and status_report['ts_event_regressions_observed'] == 0
                 and status_report['ts_recv_and_index_monotonic'], 'STATUS count/time evidence')
            record['statuses'] = normalize_status_facts(facts, day, enum_maps, status_report['metadata']['instrument_ids'])
            record['status_fact_count'] = len(facts)
            record['status_raw_sha_current_checked'] = True
        sources[day] = record
    registry = {'version': 'TCN-research-B-source-registry-1.0', 'registered_utc': now(),
                'sources': sources, 'metadata_bindings': bindings, 'enum_maps': enum_maps,
                'raw_mbo_or_native_itch_redecoded': False,
                'status_timelines_reused_from_verified_execution': True,
                'absent_adapter_anomaly_counter_zero_basis': 'Frozen adapter source and accepted complete normal return; Counter stores only occurred anomalies',
                'source_recv_scope': 'Prior accepted live MBO processing; initialization R is skipped and emits no sample; native TRAIN lacks ts_recv',
                'source_control_verified_for_engineering': False}
    write_json(REGISTRY, registry)
    proposal_path = ROOT / 'proposals/tcn_screening_rules_v3_draft_20261011.json'
    proposal = read_json(proposal_path)
    protocol = {'version': 'TCN-research-data-protocol-B-1.0', 'registered_utc': now(),
                'status': 'adopted_for_P1_screening_engineering_acceptance',
                'authorization_quote': '帮我完成第1步', 'user_direction_accepted': True,
                'scope': 'Research registration and ten-day real screening only; no labels/norm/model/training',
                'parent_protocol': str(ROOT / 'configs/tcn_data_protocol_v2.json'),
                'parent_protocol_sha256': sha(ROOT / 'configs/tcn_data_protocol_v2.json'),
                'proposal_path': str(proposal_path), 'proposal_sha256': sha(proposal_path),
                'source_registry_path': str(REGISTRY), 'source_registry_sha256': sha(REGISTRY),
                'dataset': 'XNAS.ITCH', 'symbol': 'AAPL', 'date_roles': ROLES,
                'research_mode': copy.deepcopy(proposal['modes']['B_calendar_observed_research']),
                'samples': copy.deepcopy(proposal['samples']),
                'timezone_profile': {'version': PROFILE, 'timezone_method': 'fixed_offset_target_dates',
                                     'rule_location': 'America/New_York', 'offset_seconds': -18000,
                                     'IANA_database_loaded': False, 'allowed_dates': list(ROLES),
                                     'UTC_screen_start': '14:35:00', 'UTC_screen_end_exclusive': '20:55:00',
                                     'basis': 'Registered December 2025 dates are EST; integer nanoseconds; unknown dates refused'},
                'public_sources': [{'url': 'https://www.nasdaqtrader.com/content/technicalsupport/2025tradingcalendar.pdf', 'verified_on': '2026-10-14', 'scope': '2025 calendar target dates, normal planned sessions'},
                                   {'url': 'https://www.nasdaq.com/market-activity/stock-market-holiday-schedule', 'verified_on': '2026-10-14', 'scope': 'General regular session 09:30-16:00; 2026 holiday table not used for 2025 dates'},
                                   {'url': 'https://www.nist.gov/pml/time-and-frequency-division/local-time-faqs', 'verified_on': '2026-10-14', 'scope': 'Eastern standard time UTC-5 and first-Sunday-November DST end'}],
                'implementation_bindings': [small_binding(Path(__file__)), small_binding(Path(__file__).resolve().parents[2]/'software/data/tcn_research_screening_core.py'),
                                            small_binding(Path(__file__).resolve().parents[2]/'software/data/tcn_screening_reference.py')],
                'result_wording': '基于当前可观察数据和既定筛选规则的表现',
                'unobserved_halt_coverage_risk_accepted': True, 'source_control_verified': False,
                'coverage_unknown': True, 'original_flag64_cleared': False,
                'labels_computed': False, 'normalization_computed': False, 'training_ready': False,
                'sealed_dates': NEW[2:], 'no_future_vendor_reply_or_fix_required': True}
    protocol['research_mode']['enabled'] = True
    protocol['research_mode']['recommended_priority'] = 1
    write_json(CFG, protocol)
    write_json(OUT / 'registration_receipt.json', {'registered_utc': now(), 'protocol_path': str(CFG),
               'protocol_sha256': sha(CFG), 'source_registry_sha256': sha(REGISTRY),
               'protected_small_file_count': len(protected), 'authorized_step': 'P1',
               'records_full_SHA_not_yet_rechecked': True, 'training_ready': False})
    print(json.dumps({'registration': 'completed', 'source_days': len(sources), 'protocol_sha256': sha(CFG)}, ensure_ascii=True), flush=True)


def admit():
    receipt = read_json(OUT / 'registration_receipt.json')
    need(sha(CFG) == receipt['protocol_sha256'], 'Registered protocol drift')
    protocol = read_json(CFG)
    need(protocol['date_roles'] == ROLES and protocol['research_mode']['enabled']
         and protocol['unobserved_halt_coverage_risk_accepted'], 'Research admission mismatch')
    need(sha(REGISTRY) == protocol['source_registry_sha256'], 'Source registry drift')
    registry = read_json(REGISTRY)
    for binding in protocol['implementation_bindings'] + registry['metadata_bindings']:
        verify_binding(binding)
    need(set(registry['sources']) == set(ROLES), 'Missing source day')
    for day, source in registry['sources'].items():
        need(source['day'] == day and source['role'] == ROLES[day], 'Source role drift')
        for binding in source['metadata_bindings']:
            verify_binding(binding)
    return protocol, registry


def run():
    started, timer = now(), time.perf_counter()
    protocol, registry = admit()
    need(shutil.disk_usage(ROOT).free > 1024**3, 'Workspace lacks screening output space')
    import numpy as np
    from software.data.tcn_research_screening_core import RECORD_DTYPE, REASON_BITS, screen_array
    from software.data.tcn_screening_reference import Status
    run_dir = OUT / ('real_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
    run_dir.mkdir(exist_ok=False)
    results, ranges_all = {}, []
    for day, source in registry['sources'].items():
        day_timer = time.perf_counter()
        p = Path(source['records_path']).resolve()
        need(p.is_relative_to(DATA_ROOT) and p.name == 'records.bin', 'Record read root refused')
        signature = (p.stat().st_size, p.stat().st_mtime_ns)
        need(signature[0] == source['record_bytes'], 'Record bytes changed')
        print(json.dumps({'day': day, 'phase': 'verify_source_SHA_before'}, ensure_ascii=True), flush=True)
        before = sha(p)
        need(before == source['records_sha256'], 'Source records changed before screening: ' + day)
        records = np.memmap(p, dtype=RECORD_DTYPE, mode='r')
        try:
            need(len(records) == source['rows'], 'Record count changed')
            kwargs = {'day': day, 'source_kind': source['source_kind'], 'chunk_size': 100_000}
            if source['source_kind'] == 'native':
                kwargs['native_segments'] = read_json(source['segments_path'])
            else:
                origin = midnight_est_ns(day)
                need(int(records['timestamp_ns'][0]) + origin == source['first_sample_event_unix_ns']
                     and int(records['timestamp_ns'][-1]) + origin == source['last_sample_event_unix_ns'],
                     'Engineering local clock mismatch')
                kwargs['adapter_version'] = source['adapter_version']
                kwargs['statuses'] = [Status(**{k: v for k, v in s.items() if k in
                                      ('day', 'ts_event', 'action', 'reason', 'is_trading', 'is_quoting', 'ssr', 'symbol')})
                                      for s in source['statuses']]
            print(json.dumps({'day': day, 'phase': 'screen_all_original_rows', 'rows': len(records)}, ensure_ascii=True), flush=True)
            result = screen_array(records, **kwargs)
            masks, ranges, summary = result['reason_masks'], result['ranges'], result['summary']
            need(sum(r['record_count'] for r in ranges) == int(np.count_nonzero(masks == 0)), 'Eligible row/range identity')
            need(sum(r['window_count'] for r in ranges) == summary['window_count'], 'Window sum identity')
            previous_stop = 0
            for r in ranges:
                start, stop = r['start_row'], r['stop_row']
                need(previous_stop <= start < stop <= len(records) and bool(np.all(masks[start:stop] == 0)), 'Non-contiguous or rejected range')
                need(r['record_count'] == stop - start and r['window_count'] == max(0, stop - start - 149), '150-row footprint arithmetic')
                bound = {**r, 'date': day, 'role': source['role'], 'records_path': str(p),
                         'records_sha256': before, 'source_control_verified': source['source_control_verified'],
                         'coverage_unknown': source['coverage_unknown'], 'protocol_sha256': sha(CFG)}
                if r['window_count']:
                    bound['endpoint_start'] = start + 99
                    bound['endpoint_stop_exclusive'] = stop - 50
                else:
                    bound['endpoint_start'] = None
                    bound['endpoint_stop_exclusive'] = None
                ranges_all.append(bound)
                previous_stop = stop
            day_folder = run_dir / day
            day_folder.mkdir()
            mask_path = day_folder / 'reject_reasons.u32'
            masks.astype('<u4', copy=False).tofile(mask_path)
            write_json(day_folder / 'ranges.json', ranges)
        finally:
            records._mmap.close()
        print(json.dumps({'day': day, 'phase': 'verify_source_SHA_after'}, ensure_ascii=True), flush=True)
        after = sha(p)
        need(before == after and signature == (p.stat().st_size, p.stat().st_mtime_ns), 'Source mutation during screening')
        for b in source['metadata_bindings']:
            verify_binding(b)
        summary.update({'source_records_path': str(p), 'source_sha256_before': before,
                        'source_sha256_after': after, 'source_signature_unchanged': True,
                        'role': source['role'], 'elapsed_seconds': time.perf_counter() - day_timer,
                        'mask_path': str(mask_path), 'mask_sha256': sha(mask_path),
                        'ranges_path': str(day_folder / 'ranges.json'),
                        'ranges_sha256': sha(day_folder / 'ranges.json'),
                        'protocol_sha256': sha(CFG), 'source_registry_sha256': sha(REGISTRY)})
        summary['research_window_acceptance_passed'] = research_window_acceptance(summary)
        summary['p1_acceptance_problems'] = [] if summary['window_count'] > 0 else ['zero_research_windows']
        write_json(day_folder / 'screening.json', summary)
        results[day] = summary
        print(json.dumps({'day': day, 'phase': 'day_complete', 'eligible_records': summary['eligible_records'],
                          'windows': summary['window_count'], 'day_problems': summary['day_problem_codes']}, ensure_ascii=True), flush=True)
    admit()  # Verify frozen source/config/metadata bindings again after processing.
    protected = read_json(OUT / 'protected_before.json')
    need(all(sha(ROOT / p) == digest for p, digest in protected.items()), 'Historical small file changed')
    all_pass = all(r['research_window_acceptance_passed'] for r in results.values())
    manifest = {'version': 'TCN-research-B-screened-manifest-1.0', 'completed_utc': now(),
                'protocol_path': str(CFG), 'protocol_sha256': sha(CFG),
                'source_registry_path': str(REGISTRY), 'source_registry_sha256': sha(REGISTRY),
                'date_roles': ROLES, 'sources': registry['sources'], 'ranges': ranges_all,
                'sample_counts': {role: sum(r['window_count'] for r in results.values() if r['role'] == role)
                                  for role in set(ROLES.values())},
                'unique_eligible_records_by_role': {role: sum(r['eligible_records'] for r in results.values() if r['role'] == role)
                                                   for role in set(ROLES.values())},
                'engineering_screening_passed': all_pass, 'research_data_index_ready': all_pass,
                'source_control_verified_for_new_days': False, 'coverage_unknown_for_new_days': True,
                'source_control_evidence_difference_preserved': True, 'original_flag64_cleared': False,
                'record_schema_bytes': 352, 'reject_mask_schema': {'dtype': '<u4', 'reason_bits': REASON_BITS,
                    'original_index_semantics': 'offset in unchanged source file; zero mask eligible, ranges define segment identity'},
                'normalization': None, 'train_class_counts': None, 'labels_computed': False,
                'sealed_dates': NEW[2:], 'sealed_labels_or_scores_opened': False,
                'model_forwards': 0, 'optimizer_updates': 0, 'training_runs': 0,
                'training_ready': False, 'pending_step2': 'TRAIN-only norm/counts and real entry/runtime/evaluation acceptance'}
    write_json(run_dir / 'manifest.json', manifest)
    write_json(run_dir / 'execution_receipt.json', {'started_utc': started, 'completed_utc': now(),
               'elapsed_seconds': time.perf_counter() - timer, 'status': 'screening_passed' if all_pass else 'screening_rejected',
               'all_ten_sources_scanned': True, 'total_source_records': sum(r['total_records'] for r in results.values()),
               'total_eligible_records': sum(r['eligible_records'] for r in results.values()),
               'total_windows': sum(r['window_count'] for r in results.values()),
               'per_day': results, 'protocol_sha256': sha(CFG), 'source_registry_sha256': sha(REGISTRY),
               'manifest_path': str(run_dir / 'manifest.json'), 'manifest_sha256': sha(run_dir / 'manifest.json'),
               'protected_small_files_unchanged': True, 'source_records_and_flag64_unchanged': True,
               'status_decode_calls': 0, 'raw_MBO_reconstruction_runs': 0, 'E_drive_writes': 0,
               'labels_computed': False, 'normalization_computed': False,
               'model_forwards': 0, 'optimizer_updates': 0, 'training_ready': False})
    pointer_path = OUT / 'screened_latest.json'
    need(not pointer_path.exists(), 'Completion pointer already exists; do not overwrite prior real run')
    write_json(pointer_path, {'manifest_path': str(run_dir / 'manifest.json'),
                             'manifest_sha256': sha(run_dir / 'manifest.json'),
                             'execution_receipt_path': str(run_dir / 'execution_receipt.json'),
                             'execution_receipt_sha256': sha(run_dir / 'execution_receipt.json'),
                             'engineering_screening_passed': all_pass, 'training_ready': False})
    print(json.dumps({'status': 'passed' if all_pass else 'rejected', 'output': str(run_dir),
                      'total_windows': sum(r['window_count'] for r in results.values()),
                      'elapsed_seconds': time.perf_counter() - timer}, ensure_ascii=True), flush=True)
    need(all_pass, 'A required day has no accepted research windows or a source integrity issue')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'run', 'check'))
    command = parser.parse_args().command
    if command == 'register':
        register()
    elif command == 'run':
        run()
    else:
        admit()
        print('Research configuration and metadata admission passed; no record scan or training.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        error_path = OUT / ('error_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f') + '.json')
        write_json(error_path, {'time_utc': now(), 'error_type': type(exc).__name__,
                   'error': str(exc), 'training_executed': False})
        raise
