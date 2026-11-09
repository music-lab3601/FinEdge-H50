"""Independent stdlib audit of saved P1 metadata and D-only rejection masks.

Never opens E, records/raw/model files, imports NumPy/ML or replays screening.
Saved source before/after hashes are evidence inspected, not hashes remeasured.
"""
from array import array
from collections import Counter
from datetime import date, datetime, timezone
import argparse
import hashlib
import json
import ntpath
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/tcn_research_screening_p1_2026-10-14'
CFG = ROOT / 'configs/tcn_research_protocol_b_v1.json'
ROLES = {**{f'2025-12-{d:02d}': 'train' for d in range(8, 13)},
         '2025-12-15': 'tuning_validation', '2025-12-16': 'tuning_validation',
         '2025-12-17': 'comparison_validation', '2025-12-18': 'final_test',
         '2025-12-19': 'final_test'}
SEALED = ['2025-12-17', '2025-12-18', '2025-12-19']
BITS = {'day_quarantine': 2, 'unbound_engineering_provenance': 16,
        'invalid_event_timestamp': 2048, 'sequence_non_increasing': 4096,
        'event_time_regression': 8192, 'unexpected_row_bits': 256,
        'native_control_or_quality_flag': 4, 'native_segment': 8,
        'missing_unverified_source_flag': 32, 'engineering_source_segment': 64,
        'engineering_quality_flag': 128, 'insufficient_depth': 512,
        'invalid_book_geometry': 1024, 'outside_research_clock': 1,
        'equal_status_timestamp': 16384, 'sticky_control_quarantine': 65536,
        'observed_state_blocked': 32768}
START = (9 * 3600 + 35 * 60) * 10**9
STOP = (15 * 3600 + 55 * 60) * 10**9
DAY_NS = 86400 * 10**9


def stamp():
    return datetime.now(timezone.utc).isoformat()


def safe(path, *, mask=False):
    """Reject non-workspace paths lexically before resolving or stat/open."""
    raw = str(path)
    norm, base = ntpath.normcase(ntpath.normpath(raw)), ntpath.normcase(str(ROOT))
    if not (norm == base or norm.startswith(base + '\\')):
        raise ValueError('Audit refuses non-workspace path: ' + raw)
    p = Path(raw).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Audit refuses path outside workspace after resolution')
    if mask:
        if p.name != 'reject_reasons.u32' or not p.is_relative_to(OUT):
            raise ValueError('Only new P1 reject mask binaries are allowed')
    elif p.suffix.lower() not in {'.py', '.json', '.jsonl', '.txt', '.log', '.md'}:
        raise ValueError('Audit refuses non-metadata/non-source file')
    return p


def read(path):
    p = safe(path)
    if p.stat().st_size > 2 * 1024**2:
        raise ValueError('Metadata exceeds 2 MiB audit bound')
    return json.loads(p.read_text(encoding='utf-8'))


def sha(path):
    p = safe(path)
    if p.stat().st_size > 2 * 1024**2:
        raise ValueError('Audit refuses large metadata/source hash')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lexical(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def is_e_metadata(binding):
    return ntpath.splitdrive(binding['path'])[0].lower() == 'e:'


class Audit:
    def __init__(self):
        self.checks = []
        self.skipped_e = []
        self.by_day = {}

    def check(self, condition, label):
        passed = bool(condition)
        self.checks.append({'check': label, 'passed': passed})
        if not passed:
            raise ValueError(label)

    def binding(self, binding):
        if is_e_metadata(binding):
            # Only inspect literal saved fields. Absolutely no E resolution/stat.
            self.check(re.fullmatch('[0-9a-f]{64}', binding['sha256']) is not None,
                       'Saved E metadata SHA syntax: ' + binding['path'])
            self.check(type(binding['bytes']) is int and binding['bytes'] >= 0,
                       'Saved E metadata byte count: ' + binding['path'])
            self.skipped_e.append(binding)
            return
        p = safe(binding['path'])
        self.check(sha(p) == binding['sha256'], 'Current D binding SHA: ' + str(p))
        self.check(p.stat().st_size == binding['bytes'], 'Current D binding size: ' + str(p))

    def mask(self, path, ranges, expected_rows):
        p = safe(path, mask=True)
        self.check(p.stat().st_size == expected_rows * 4, 'Mask bytes / original source rows: ' + p.parent.name)
        digest, counts, values = hashlib.sha256(), Counter(), array('I')
        self.check(values.itemsize == 4, 'stdlib uint32 width')
        # Exactly one read and one Counter classification for each saved mask.
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 262144), b''):
                self.check(len(block) % 4 == 0, 'Mask complete uint32 block: ' + p.parent.name)
                digest.update(block)
                part = array('I')
                part.frombytes(block)
                if sys.byteorder != 'little':
                    part.byteswap()
                counts.update(part)
                values.extend(part)
        self.check(len(values) == expected_rows, 'Mask array original length: ' + p.parent.name)
        known = sum(BITS.values())
        self.check(all(v & ~known == 0 for v in counts), 'No unknown reject bits: ' + p.parent.name)
        overlaps = {k: sum(n for v, n in counts.items() if v & bit) for k, bit in BITS.items()}
        primary = dict.fromkeys(BITS, 0)
        for v, n in counts.items():
            if not v:
                continue
            for k, bit in BITS.items():
                if v & bit:
                    primary[k] += n
                    break
        stop, eligible, windows = 0, 0, 0
        for ordinal, r in enumerate(ranges, 1):
            a, b = r['start_row'], r['stop_row']
            self.check(type(a) is int and type(b) is int and stop <= a < b <= expected_rows,
                       f'Original nonoverlapping offset range {p.parent.name}/{ordinal}')
            self.check(r['research_segment_id'] == ordinal, f'Unique ordered research segment {p.parent.name}/{ordinal}')
            self.check(r['record_count'] == b - a and r['window_count'] == max(0, b - a - 149),
                       f'Independent 150-row footprint formula {p.parent.name}/{ordinal}')
            self.check(values[a:b].count(0) == b - a, f'Every range row mask zero {p.parent.name}/{ordinal}')
            stop, eligible, windows = b, eligible + b - a, windows + max(0, b - a - 149)
        self.check(eligible == counts[0], 'Ranges cover all and only eligible original rows: ' + p.parent.name)
        self.check(sum(primary.values()) == expected_rows - eligible,
                   'Primary counts partition rejects: ' + p.parent.name)
        return {'sha256': digest.hexdigest(), 'mask_combination_counts': dict(sorted(counts.items())),
                'eligible': eligible, 'windows': windows, 'reason_counts': overlaps,
                'primary_reason_counts': primary}

    def run(self):
        pointer = read(OUT / 'screened_latest.json')
        self.check(pointer['engineering_screening_passed'] is True and pointer['training_ready'] is False,
                   'Completion pointer is engineering-only success')
        manifest_path = safe(pointer['manifest_path'])
        receipt_path = safe(pointer['execution_receipt_path'])
        self.check(manifest_path.parent == receipt_path.parent and manifest_path.parent.parent == OUT,
                   'One contained real output directory')
        self.check(sha(manifest_path) == pointer['manifest_sha256'], 'Completion manifest SHA')
        self.check(sha(receipt_path) == pointer['execution_receipt_sha256'], 'Completion execution receipt SHA')
        manifest, receipt = read(manifest_path), read(receipt_path)
        registration = read(OUT / 'registration_receipt.json')
        protocol = read(CFG)
        registry_path = safe(protocol['source_registry_path'])
        registry = read(registry_path)
        protocol_sha, registry_sha = sha(CFG), sha(registry_path)
        self.check(protocol_sha == registration['protocol_sha256'] == manifest['protocol_sha256'] == receipt['protocol_sha256'],
                   'Registered/current/run protocol identity')
        self.check(registry_sha == registration['source_registry_sha256'] == protocol['source_registry_sha256']
                   == manifest['source_registry_sha256'] == receipt['source_registry_sha256'], 'Registered/current/run registry identity')
        self.check(lexical(manifest['protocol_path']) == lexical(CFG) and lexical(manifest['source_registry_path']) == lexical(registry_path),
                   'Manifest protocol/registry aliases')
        self.check(receipt['manifest_path'] == str(manifest_path) and receipt['manifest_sha256'] == sha(manifest_path),
                   'Receipt manifest alias and identity')
        self.check(manifest['date_roles'] == protocol['date_roles'] == ROLES and set(registry['sources']) == set(ROLES),
                   'Exact ten dates and frozen roles')
        self.check(manifest['sources'] == registry['sources'], 'Manifest source aliases exactly match registry')
        self.check(manifest['sealed_dates'] == protocol['sealed_dates'] == SEALED, 'Sealed comparison/final dates preserved')
        self.check(protocol['samples']['required_footprint_records'] == 150
                   and protocol['samples']['window_length'] == 100 and protocol['samples']['horizon_records'] == 50,
                   '100 input + 50 future complete original footprint')
        tz = protocol['timezone_profile']
        self.check(tz['timezone_method'] == 'fixed_offset_target_dates' and tz['offset_seconds'] == -18000
                   and tz['IANA_database_loaded'] is False and tz['allowed_dates'] == list(ROLES), 'Honest fixed-EST whitelist profile')
        self.check(tz['UTC_screen_start'] == '14:35:00' and tz['UTC_screen_end_exclusive'] == '20:55:00', 'Frozen EST UTC clock bounds')
        self.check(protocol['research_mode']['enabled'] is True and protocol['unobserved_halt_coverage_risk_accepted'] is True
                   and protocol['source_control_verified'] is False and protocol['coverage_unknown'] is True,
                   'Research route and coverage limitation retained')
        for obj, name in [(protocol, 'protocol'), (manifest, 'manifest'), (receipt, 'execution')]:
            self.check(obj['training_ready'] is False and obj['labels_computed'] is False, name + ' has no training/labels')
        self.check(protocol['normalization_computed'] is False and receipt['normalization_computed'] is False
                   and manifest['normalization'] is None and manifest['train_class_counts'] is None,
                   'No new normalization or class labels')
        self.check(manifest['sealed_labels_or_scores_opened'] is False, 'Sealed labels/scores unopened')
        self.check(manifest['original_flag64_cleared'] is False and protocol['original_flag64_cleared'] is False
                   and receipt['source_records_and_flag64_unchanged'] is True, 'Original source flags retained')
        self.check(manifest['engineering_screening_passed'] is True and manifest['research_data_index_ready'] is True
                   and receipt['status'] == 'screening_passed', 'Ten-day research index completion')
        self.check(manifest['source_control_verified_for_new_days'] is False and manifest['coverage_unknown_for_new_days'] is True,
                   'New-day control source still unverified')
        self.check(manifest['reject_mask_schema']['dtype'] == '<u4' and manifest['reject_mask_schema']['reason_bits'] == BITS,
                   'Independent uint32 reject schema/order')
        self.check(list(manifest['reject_mask_schema']['reason_bits']) == list(BITS), 'Primary reason priority order')
        for key in ['model_forwards', 'optimizer_updates', 'training_runs']:
            self.check(manifest[key] == 0, 'Manifest no-model scope: ' + key)
        for key in ['model_forwards', 'optimizer_updates', 'status_decode_calls', 'raw_MBO_reconstruction_runs', 'E_drive_writes']:
            self.check(receipt[key] == 0, 'Execution no-decoding/model/E-write scope: ' + key)
        self.check(registry['raw_mbo_or_native_itch_redecoded'] is False
                   and registry['status_timelines_reused_from_verified_execution'] is True, 'Registered source evidence reused, not redecoded')
        for binding in protocol['implementation_bindings'] + registry['metadata_bindings']:
            self.binding(binding)
        self.check(sha(protocol['proposal_path']) == protocol['proposal_sha256'], 'Draft history identity retained')
        protected = read(OUT / 'protected_before.json')
        self.check(len(protected) == registration['protected_small_file_count'], 'Historical protection list count')
        for relative, expected in protected.items():
            self.check(sha(ROOT / relative) == expected, 'Historical protected D small file: ' + relative)
        # These originals remain historical; their pending handoff wording is not a failure.
        self.check((OUT / 'previous_AGENTS.md').is_file() and (OUT / 'previous_PROGRESS.md').is_file(), 'Original handoff snapshots retained')
        upstream_path = ROOT / 'artifacts/tcn_data_v2_2026-10-09/manifest.json'
        upstream = read(upstream_path)
        aggregate_ranges, counts_role, records_role = [], Counter(), Counter()
        self.check(set(receipt['per_day']) == set(ROLES), 'Receipt covers each required day')
        for day, source in registry['sources'].items():
            summary_path = manifest_path.parent / day / 'screening.json'
            summary = read(summary_path)
            self.check(summary == receipt['per_day'][day], 'Per-day receipt/summary exact identity: ' + day)
            self.check(source['day'] == summary['day'] == day and source['role'] == summary['role'] == ROLES[day], 'Day/role binding: ' + day)
            kind = 'native' if ROLES[day] == 'train' else 'engineering'
            self.check(source['source_kind'] == summary['source_kind'] == kind, 'Day source kind: ' + day)
            self.check(summary['source_control_verified'] == source['source_control_verified'] == (kind == 'native')
                       and summary['coverage_unknown'] == source['coverage_unknown'] == (kind == 'engineering'), 'Source evidence difference: ' + day)
            self.check(summary['source_sha256_before'] == summary['source_sha256_after'] == source['records_sha256']
                       and summary['source_signature_unchanged'] is True, 'Saved before/after source SHA evidence only: ' + day)
            self.check(lexical(summary['source_records_path']) == lexical(source['records_path'])
                       and source['record_bytes'] == 352 * source['rows'], 'Original source path/bytes alias: ' + day)
            self.check(summary['protocol_sha256'] == protocol_sha and summary['source_registry_sha256'] == registry_sha,
                       'Per-day frozen identities: ' + day)
            self.check(summary['total_records'] == source['rows'], 'Original per-day denominator: ' + day)
            self.check(summary['day_problem_codes'] == [] and summary['p1_acceptance_problems'] == []
                       and summary['research_window_acceptance_passed'] is True and summary['window_count'] > 0,
                       'Per-required-day nonzero window admission: ' + day)
            for key in ['invalid_timestamp_count', 'sequence_anomaly_count', 'event_regression_count']:
                self.check(summary[key] == 0, 'Saved actual integrity counter: ' + day + '/' + key)
            self.check(summary['labels_computed'] is False and summary['normalization_computed'] is False
                       and summary['training_ready'] is False and summary['accuracy'] is None, 'Unmeasured quality remains null: ' + day)
            for binding in source['metadata_bindings']:
                self.binding(binding)
            ranges_path = safe(summary['ranges_path'])
            mask_path = safe(summary['mask_path'], mask=True)
            self.check(ranges_path == summary_path.parent / 'ranges.json'
                       and mask_path == summary_path.parent / 'reject_reasons.u32', 'Contained day sidecar aliases: ' + day)
            self.check(sha(ranges_path) == summary['ranges_sha256'], 'Per-day ranges SHA: ' + day)
            ranges = read(ranges_path)
            counted = self.mask(mask_path, ranges, source['rows'])
            self.check(counted['sha256'] == summary['mask_sha256'], 'Single-pass sidecar SHA: ' + day)
            self.check(counted['eligible'] == summary['eligible_records'] and counted['windows'] == summary['window_count']
                       and summary['rejected_records'] == source['rows'] - counted['eligible'], 'Independent mask/range summary counts: ' + day)
            self.check(summary['segment_count'] == len(ranges) and summary['reason_counts_overlap'] is True, 'Segment and overlap declarations: ' + day)
            for field in ['reason_counts', 'primary_reason_counts']:
                self.check(set(summary[field]) == set(BITS), 'Every reject reason reported: ' + day + '/' + field)
                for key in BITS:
                    self.check(summary[field][key] == counted[field][key], 'Independent reject count: ' + day + '/' + field + '/' + key)
            if kind == 'native':
                accepted = [r for r in upstream['ranges'] if r['date'] == day]
                self.check(bool(accepted) and source['records_sha256'] == upstream['sources'][day]['records_sha256'],
                           'Original strict accepted source metadata: ' + day)
                for r in ranges:
                    self.check(any(a['segment_start'] <= r['start_row'] < r['stop_row'] <= a['segment_stop'] for a in accepted),
                               'Research range stays inside one original strict segment: ' + day)
                control_support = {'method': 'Original D accepted strict segment metadata containment',
                                   'E_segments_reopened': False, 'MBO_or_control_events_replayed': False}
                self.check(summary['status_records'] == 0 and summary['status_timestamp_groups'] == 0, 'Native gate not substituted with status: ' + day)
            else:
                statuses = source['statuses']
                self.check(len(statuses) == source['status_fact_count'] == summary['status_records']
                           and summary['status_timestamp_groups'] == len({s['ts_event'] for s in statuses}), 'STATUS full timeline count: ' + day)
                origin = ((date.fromisoformat(day) - date(1970, 1, 1)).days * 86400 + 5 * 3600) * 10**9
                for ordinal, s in enumerate(statuses, 1):
                    self.check(s['day'] == day and s['symbol'] == 'AAPL' and s['source_record_ordinal'] == ordinal
                               and s['source_publisher_id'] == 2 and s['source_instrument_id'] == 38,
                               'STATUS observed identity/ordinal: ' + day + '/' + str(ordinal))
                    self.check(type(s['ts_event']) is int and 0 <= s['ts_event'] < DAY_NS
                               and s['ts_event'] + origin == s['source_ts_event'], 'Independent integer EST normalization: ' + day + '/' + str(ordinal))
                self.check(all(a['ts_event'] <= b['ts_event'] and a['source_ts_recv'] <= b['source_ts_recv']
                               for a, b in zip(statuses, statuses[1:])), 'STATUS original order retained: ' + day)
                # Actual accepted five days have no observed control inside the clock.
                inside = [s for s in statuses if START <= s['ts_event'] < STOP]
                self.check(not inside and statuses[-1]['ts_event'] < START
                           and statuses[-1]['action'] == 'TRADING' and statuses[-1]['reason'] == 'NONE'
                           and statuses[-1]['is_trading'] == statuses[-1]['is_quoting'] == 'Y', 'Observed pre-clock TRADING restoration / no inside-clock controls: ' + day)
                control_support = {'method': 'Normalized saved STATUS facts: latest TRADING Y/Y before 09:35; no observed control inside selected clock',
                                   'unobserved_control_risk_remains': True, 'raw_join_or_timestamp_filter_reexecuted': False,
                                   'inside_clock_observed_controls': len(inside)}
            for r in ranges:
                bound = {**r, 'date': day, 'role': ROLES[day], 'records_path': source['records_path'],
                         'records_sha256': source['records_sha256'], 'source_control_verified': source['source_control_verified'],
                         'coverage_unknown': source['coverage_unknown'], 'protocol_sha256': protocol_sha,
                         'endpoint_start': r['start_row'] + 99 if r['window_count'] else None,
                         'endpoint_stop_exclusive': r['stop_row'] - 50 if r['window_count'] else None}
                aggregate_ranges.append(bound)
            counts_role[ROLES[day]] += counted['windows']
            records_role[ROLES[day]] += counted['eligible']
            self.by_day[day] = {**counted, 'source_records': source['rows'], 'segment_count': len(ranges),
                                'control_boundary_metadata_support': control_support}
        self.check(manifest['ranges'] == aggregate_ranges, 'Global range aliases/footprint/endpoints/roles exactly reconstructed')
        self.check(manifest['sample_counts'] == dict(counts_role), 'Independent role window sums')
        self.check(manifest['unique_eligible_records_by_role'] == dict(records_role), 'Independent unique role row sums')
        self.check(receipt['total_source_records'] == sum(s['rows'] for s in registry['sources'].values())
                   and receipt['total_eligible_records'] == sum(records_role.values())
                   and receipt['total_windows'] == sum(counts_role.values()), 'Independent all-ten-day totals')
        self.check(receipt['all_ten_sources_scanned'] is True and receipt['protected_small_files_unchanged'] is True,
                   'Original actual execution / protected history declarations')
        final_tests = read(OUT / 'screening_p1_tests_bound_final_receipt.json')
        self.check(final_tests['test_count'] == 42 and all(final_tests[k] == 0 for k in ('exit_code', 'failures', 'errors', 'skipped')),
                   'Final runner synthetic test outcome')
        self.check(final_tests['runner_sha256'] == sha(ROOT / 'scripts/tcn_research_screening_p1.py')
                   and final_tests['test_sha256'] == sha(ROOT / 'tests/test_tcn_research_screening_p1.py'), 'Final runner/test source bindings')
        self.check(final_tests['real_register_or_run_calls'] == 0 and final_tests['ML_imported'] is False
                   and final_tests['model_or_training_executed'] is False, 'Synthetic test scope distinct from real screening')
        core_log = safe(OUT / 'screening_core_tests_after_import_fix.log').read_text(encoding='utf-8')
        self.check(re.search(r'Ran 48 tests in ', core_log) is not None and core_log.rstrip().endswith('OK'), 'Final core48 synthetic log passed')
        core_binding_path = OUT / 'screening_core_tests_bound_final_receipt.json'
        self.check(core_binding_path.is_file(), 'Actual final bound core execution receipt exists')
        core_tests = read(core_binding_path)
        self.check(core_tests['test_count'] == 48
                   and all(core_tests[k] == 0 for k in ('exit_code', 'failures', 'errors', 'skipped')),
                   'Core final receipt outcome')
        self.check(core_tests['core_sha256_before'] == core_tests['core_sha256_after']
                   == sha(ROOT / 'scripts/tcn_research_screening_core.py')
                   and core_tests['test_sha256_before'] == core_tests['test_sha256_after']
                   == sha(ROOT / 'tests/test_tcn_research_screening_core.py')
                   and core_tests['source_identity_unchanged'] is True, 'Core final receipt frozen before/after source bindings')
        self.check(sha(core_tests['log_path']) == core_tests['log_sha256'], 'Actual bound core test log SHA')
        self.check(sha(core_tests['actual_command'][1]) == core_tests['execution_helper_sha256'], 'Actual core test helper identity')
        self.check(core_tests['real_records_or_E_access'] is False and core_tests['runner_suite_reexecuted'] is False
                   and core_tests['model_runtime_imported'] is False and core_tests['model_or_training_executed'] is False,
                   'Additional core test actual scope distinguished from audit')
        core_test_binding = 'additional actual 48-test execution receipt before/after source hashes checked; historical earlier log preserved'
        return {'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                'protocol_sha256': protocol_sha, 'registry_sha256': registry_sha,
                'per_day': self.by_day, 'sample_counts': dict(counts_role),
                'eligible_records_by_role': dict(records_role), 'total_windows': sum(counts_role.values()),
                'E_metadata_bindings_inspected_without_open': self.skipped_e,
                'core_test_source_binding_status': core_test_binding,
                'limitations': ['Original E source SHA only inspected in saved before/after receipts, not rehashed by auditor.',
                                'Source record geometry, sequence and event timestamps are not reread; actual scan is attested by frozen implementation plus saved execution.',
                                'Control boundaries checked from saved metadata; no raw control join or native ordering is replayed.',
                                'Research eligibility does not prove complete unobserved H/h/W coverage.'],
                'records_or_raw_or_models_opened': False, 'E_reads_or_writes': 0,
                'sidecar_mask_classification_passes_per_day': 1,
                'ML_or_numpy_imported': False, 'screening_core_calls': 0,
                'labels_or_normalizer_computed': False, 'model_forwards': 0,
                'optimizer_updates': 0, 'training_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default=str(OUT / 'readonly_audit_final'))
    args = parser.parse_args()
    prefix = Path(args.output_prefix)
    audit_path, log_path, receipt_path = (safe(str(prefix) + suffix) for suffix in ('.json', '.log', '_execution_receipt.json'))
    if any(p.exists() for p in (audit_path, log_path, receipt_path)):
        raise ValueError('Audit evidence already exists; do not repeat or overwrite')
    started, timer = stamp(), time.perf_counter()
    audit, result, error, code = Audit(), {}, None, 0
    try:
        result = audit.run()
    except Exception as exc:
        error, code = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}, 2
    completed = stamp()
    report = {'version': 'TCN-research-P1-independent-stdlib-audit-1.0', 'started_utc': started,
              'completed_utc': completed, 'status': 'passed' if not code else 'failed',
              'checks_passed': sum(c['passed'] for c in audit.checks), 'checks_total': len(audit.checks),
              'checks': audit.checks, 'error': error, 'result': result,
              'audit_script_sha256': sha(Path(__file__)), 'E_access': 0,
              'ML_imports': 0, 'screening_core_calls': 0, 'model_forwards': 0,
              'training_ready': False}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    log = [f'Started {started}', f'Completed {completed}',
           f'Status {report["status"]}: {report["checks_passed"]}/{report["checks_total"]} checks',
           'D metadata and newly generated rejection masks only; E/raw/record/model access 0; ML/core calls 0.']
    if error:
        log.append(error['traceback'])
    with log_path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(log) + '\n')
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump({'started_utc': started, 'completed_utc': completed, 'exit_code': code,
                   'elapsed_seconds': time.perf_counter() - timer,
                   'audit_path': str(audit_path), 'audit_sha256': sha(audit_path),
                   'log_path': str(log_path), 'log_sha256': sha(log_path),
                   'script_path': str(Path(__file__).resolve()), 'script_sha256': sha(Path(__file__)),
                   'E_access': 0, 'original_source_hash_calls': 0,
                   'ML_imports': 0, 'screening_core_calls': 0, 'training_ready': False}, stream,
                  ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'audit_status': report['status'], 'checks': report['checks_total'],
                      'output': str(audit_path), 'exit_code': code}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
