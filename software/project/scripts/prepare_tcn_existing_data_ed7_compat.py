"""Metadata-only versioned registration preparation. No numerical imports."""
from pathlib import Path
import json
import shutil

import tcn_existing_data_ed7_compat_contract as c


def prepare():
    c.need(not c.STAGE.exists(), 'Compatibility preparation already exists')
    c.STAGE.mkdir()
    (c.STAGE / 'before').mkdir()
    (c.STAGE / 'prepared').mkdir()
    for relative, name in [('AGENTS.md', 'AGENTS.md'), ('PROGRESS.md', 'PROGRESS.md'),
                           ('state/tcn_existing_data_optimization_20261017.json', 'state.json'),
                           ('state/tcn_software_plan_v26.json', 'plan_v26.json'),
                           ('artifacts/tcn_existing_data_optimization_2026-10-17/budget.json', 'budget.json')]:
        shutil.copyfile(c.ROOT / relative, c.STAGE / 'before' / name)
    old = c.read(c.BASE_STAGE / 'registration.json')
    history = [c.binding(p) for p in sorted(c.BASE_STAGE.rglob('*'))
               if p.is_file() and not p.is_relative_to(c.STAGE)
               and p.name != 'active_execution.json']
    protected = {str(c.safe(ref['path'])): ref
                 for ref in c.read(c.verify(old['protected_inventory_binding']))['files']}
    for ref in history + list(old['source_bindings'].values()) + [old['test_capture_binding']]:
        c.verify(ref)
        protected[str(c.safe(ref['path']))] = ref
    c.write_new(c.STAGE / 'protected_inventory.json', {
        'version': 'TCN-ED7-compat-protected-inventory-1.0',
        'created_utc': c.utc(), 'files': list(protected.values()),
        'opaque_identity_only': True,
        'scope': 'Inherited historical identity protection plus immutable original ED7 failed source/receipts/partial package. No numerical parsing, E source or sealed record streams.'})
    c.write_new(c.STAGE / 'authorization.json', {
        'version': 'TCN-ED7-versioned-compatibility-amendment-1.0',
        'created_utc': c.utc(), 'user_request': '帮我完成第7步',
        'scope': 'Necessary reversible TorchScript export compatibility repair within ED7; finite new engineering reservation, zero training.',
        'original_primary_reservation_remains_consumed': True,
        'original_primary_exit_code': 1,
        'reason': 'Installed TorchScript rejects Module.requires_grad_; use eval plus inherited frozen flags for compiled modules and per-Tensor requires_grad_(False) for loaded modules.',
        'failed_primary_operations': c.FAILED_PRIMARY_OPERATIONS,
        'primary_budget': c.PRIMARY_BUDGET, 'reload_budget': c.RELOAD_BUDGET,
        'cumulative_budget': c.CUMULATIVE_BUDGET,
        'saved_input_read_counter_scope': 'Numerical decoding of the original failed attempt saved 512000-byte normalized input file in compatibility primary only. Excludes opaque hashes/copies, new package reload/audit input decoding and small logits.',
        'training_budget_reopened': False, 'new_independent_test': False,
        'sealed17_19_stream_reads': False, 'automatic_retry': False})
    shutil.copyfile(c.BASE_STAGE / 'prepared/window_plan.json', c.STAGE / 'prepared/window_plan.json')
    reg = dict(old)
    reg.pop('test_capture_binding', None)
    reg.pop('synthetic_revision', None)
    reg.update(created_utc=c.utc(), compatibility_revision='torchscript_compatibility_v3',
        primary_budget=c.PRIMARY_BUDGET, reload_budget=c.RELOAD_BUDGET,
        failed_primary_operations=c.FAILED_PRIMARY_OPERATIONS, cumulative_budget=c.CUMULATIVE_BUDGET,
        original_registration_binding=c.binding(c.BASE_STAGE / 'registration.json'),
        initial_failed_primary_binding=c.binding(c.BASE_STAGE / 'primary_failure.json'),
        initial_primary_receipt_binding=c.binding(c.BASE_STAGE / 'primary_execution_receipt.json'),
        saved_golden_input_binding=c.binding(c.BASE_STAGE / 'package/golden/inputs.f32'),
        saved_golden_selection_binding=c.binding(c.BASE_STAGE / 'package/golden/selection.json'),
        window_plan_binding=c.binding(c.STAGE / 'prepared/window_plan.json'),
        protected_inventory_binding=c.binding(c.STAGE / 'protected_inventory.json'),
        budget_snapshot_binding=c.binding(c.STAGE / 'before/budget.json'),
        history_archive_bindings=history,
        saved_input_read_counter_scope='Original failed-attempt saved input decoded once by compat primary; excludes opaque identity/copy and new package reload/audit decoding.')
    reg['evidence_bindings'] = dict(old['evidence_bindings'], compatibility_authorization=c.binding(c.STAGE / 'authorization.json'))
    reg['package_copies'] = [dict(item) for item in old['package_copies']]
    # Source and review/test identities are filled only after all new sources settle.
    c.write_new(c.STAGE / 'registration_draft.json', reg)
    print(json.dumps({'prepared': str(c.STAGE), 'protected': len(protected), 'history_files': len(history)}))


if __name__ == '__main__':
    prepare()
