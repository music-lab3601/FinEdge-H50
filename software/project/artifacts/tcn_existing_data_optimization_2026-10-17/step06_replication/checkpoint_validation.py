"""Bounded new-checkpoint CPU deserialization audit: exactly best/latest once.

No model or optimizer is constructed; no inference, backward, optimizer update,
data input, historical checkpoint read, or RNG restoration is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require('..' not in p.parts and p.is_relative_to(ROOT), 'Outside D workspace')
    for q in (p, *p.parents):
        if q.exists():
            require(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                    'Reparse checkpoint audit input')
        if q == ROOT:
            break
    require(p.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escaped workspace')
    return p


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = local(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(path):
    p = local(path)
    require(p.stat().st_size <= 32 * 1024**2, 'Oversized checkpoint metadata')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def checked(ref):
    p = local(ref['path'])
    require(p.is_file() and sha(p) == ref['sha256'] and p.stat().st_size == ref['bytes'],
            'Checkpoint audit binding changed')
    return p


def rank(matrix):
    require(isinstance(matrix, list) and len(matrix) == 3 and all(len(row) == 3 for row in matrix), 'Bad matrix')
    require(all(type(x) is int and x >= 0 for row in matrix for x in row), 'Bad matrix counts')
    support = [sum(row) for row in matrix]; total = sum(support); require(total > 0, 'Empty matrix')
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    return (Fraction(sum(matrix[i][i] for i in range(3)), total),
            sum((Fraction(2 * matrix[i][i], support[i] + predicted[i])
                 if support[i] + predicted[i] else Fraction(0) for i in range(3)), Fraction(0)) / 3)


def replay(history):
    best = None; selected = None; wait = 0
    for i, item in enumerate(history, 1):
        require(item['epoch'] == i and item['optimizer_updates'] == 48362, 'Incomplete checkpoint epoch')
        candidate = rank(item['development_validation']['confusion_true_rows_predicted_columns'])
        improved = best is None or candidate > best; wait = 0 if improved else wait + 1
        if improved:
            best, selected = candidate, i
        require(item['improved'] is improved and item['wait'] == wait, 'Checkpoint best replay differs')
        require(i == len(history) or wait < 3, 'Checkpoint crossed patience boundary')
    return selected, wait, [[v.numerator, v.denominator] for v in best]


def inspect(cp, role, acceptance, history, registration, torch, slot):
    epoch = acceptance['selected_epoch'] if role == 'best.pt' else acceptance['epochs_completed']
    require(cp['version'] == 'TCN-ED6-complete-epoch-checkpoint-1.0'
            and cp['experiment'] == 'TCN_EDO_20261017_ED6_' + slot
            and cp['epoch'] == epoch and cp['history'] == history[:epoch], 'Checkpoint epoch/history differs')
    selected, wait, best_rank = replay(cp['history'])
    require(cp['selected_epoch'] == selected and cp['wait'] == wait and cp['best_rank'] == best_rank,
            'Checkpoint selection metadata differs')
    require(cp['identity'] == acceptance['identity'] and cp['config'] == read(checked(registration['config_binding'])),
            'Checkpoint source/config/normalization identity differs')
    require(cp['config']['slot_id'] == cp['identity']['slot_id'] == acceptance['slot_id'] == slot
            and cp['config']['seed'] == cp['identity']['seed'] == 43
            and cp['config']['learning_rate'] == .001
            and cp['identity']['selection_lock_binding'] == registration['selection_lock_binding']
            and cp['identity']['budget_prefix_snapshot_binding'] == registration['budget_prefix_snapshot_binding']
            and cp['identity']['normalization_binding'] == acceptance['normalization_binding'],
            'Checkpoint paired slot/seed/config/lock/normalization identity differs')
    require(cp['cache_binding'] == history[epoch - 1]['cache_binding']
            and cp['optimizer_attempts_charged'] == cp['optimizer_steps_completed'] == epoch * 48362,
            'Checkpoint cache/cumulative update identity differs')
    weights = cp['model']
    require(isinstance(weights, dict) and weights, 'Missing checkpoint weight tensors')
    state_h = hashlib.sha256(); parameters = 0
    for name, tensor in sorted(weights.items()):
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor)
                and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
                and torch.isfinite(tensor).all().item(), 'Invalid/nonfinite/non-CPU model tensor')
        parameters += tensor.numel()
        state_h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        state_h.update(tensor.contiguous().numpy().tobytes())
    require(parameters == 33539, 'WN checkpoint tensor parameter count differs')
    opt = cp['optimizer']; groups = opt['param_groups']; states = opt['state']
    require(isinstance(groups, list) and len(groups) == 1 and groups[0]['lr'] == .001
            and groups[0]['weight_decay'] == .0001 and groups[0]['betas'] == (.9, .999),
            'AdamW checkpoint group settings differ')
    ids = groups[0]['params']
    require(len(ids) == len(weights) and set(ids) == set(states) and len(ids) == len(set(ids)),
            'AdamW checkpoint parameter state population differs')
    for key, tensor in zip(ids, weights.values()):
        item = states[key]
        require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected AdamW state schema')
        step = item['step']
        require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.numel() == 1
                and float(step.item()) == epoch * 48362, 'AdamW per-parameter step count differs')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = item[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch.float32 and value.shape == tensor.shape
                    and torch.isfinite(value).all().item(), 'AdamW moment shape/value differs')
        require(torch.all(item['exp_avg_sq'] >= 0).item(), 'Negative AdamW second moment')
    rng = cp['rng']
    require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}, 'Checkpoint RNG population differs')
    require(isinstance(rng['python'], tuple) and len(rng['python']) == 3 and rng['python'][0] == 3,
            'Python RNG checkpoint differs')
    ns = rng['numpy']
    require(isinstance(ns, list) and len(ns) == 5 and ns[0] == 'MT19937'
            and len(ns[1]) == 624 and all(type(v) is int and 0 <= v < 2**32 for v in ns[1])
            and 0 <= ns[2] <= 624, 'NumPy RNG checkpoint differs')
    require(isinstance(rng['torch_cpu'], torch.Tensor) and rng['torch_cpu'].dtype == torch.uint8
            and rng['torch_cpu'].device.type == 'cpu' and rng['torch_cpu'].numel() > 0,
            'Torch CPU RNG checkpoint differs')
    require(isinstance(rng['torch_cuda'], list) and len(rng['torch_cuda']) >= 1
            and all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.device.type == 'cpu'
                    and v.numel() > 0 for v in rng['torch_cuda']), 'CUDA RNG snapshot differs')
    return {'checkpoint_role': role, 'passed': True, 'epoch': epoch, 'selected_epoch': selected,
            'weight_state_sha256': state_h.hexdigest(), 'parameter_numel': parameters,
            'tensor_count': len(weights), 'optimizer_parameter_states': len(states),
            'all_optimizer_steps': epoch * 48362, 'RNG_present_but_not_restored': True,
            'model_constructed': False, 'optimizer_constructed': False, 'forwards': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', choices=['control43', 'candidate43'], required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    slot = args.slot;trial = STAGE / 'trials' / slot
    started = time.perf_counter(); path = trial / 'checkpoint_validation_registration.json'
    require(sha(path) == args.registration_sha256, 'Explicit checkpoint-validation registration differs')
    reg = read(path)
    require(reg['version'] == 'TCN-ED6-bounded-checkpoint-registration-1.0'
            and reg['limits'] == {'new_checkpoint_CPU_loads': 2, 'models': 0, 'optimizers': 0,
                                  'forwards': 0, 'backwards': 0, 'optimizer_updates': 0}, 'Bounded load protocol differs')
    require(checked(reg['source_binding']) == Path(__file__), 'Checkpoint helper source differs')
    actual = read(checked(reg['training_receipt_binding']))
    require(actual['phase'] == 'training' and actual['exit_code'] == 0 and actual['entry_unchanged'] is True
            and actual['entry_before'] == actual['entry_after'] and actual['slot_id'] == slot,
            'Actual paired training success missing')
    acceptance = read(checked(actual['outputs']['training_acceptance']))
    training_registration = read(checked(reg['training_registration_binding']))
    require(reg['training_registration_binding']['sha256'] == actual['input']['sha256']
            and acceptance['identity']['registration_binding']['sha256'] == actual['input']['sha256'],
            'Actual training, acceptance and registration identity chain differs')
    require(training_registration['slot_id'] == acceptance['slot_id'] == slot
            and acceptance['selection_lock_binding'] == training_registration['selection_lock_binding']
            and acceptance['identity']['selection_lock_binding'] == training_registration['selection_lock_binding'],
            'Paired trial or pre-seed43 family lock differs')
    history = read(checked(acceptance['history_binding']))
    require(set(reg['checkpoint_bindings']) == {'best.pt', 'latest.pt'}
            and reg['checkpoint_bindings'] == acceptance['checkpoint_bindings'], 'Bounded checkpoint identities differ')
    require(not (trial / 'checkpoint_validation_reservation.json').exists()
            and not (trial / 'checkpoint_validation.json').exists(), 'Checkpoint audit already reserved')
    for name, ref in reg['checkpoint_bindings'].items():
        require(checked(ref) == trial / 'run' / name, 'Only this new paired run best/latest may be loaded')
    reservation = {'registration_binding': identity(path), 'worker_pid': os.getpid(),
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'permanent_single_use': True,
                   'limits': reg['limits']}
    with (trial / 'checkpoint_validation_reservation.json').open('x', encoding='utf-8') as stream:
        json.dump(reservation, stream, indent=2); stream.write('\n')
    import torch
    torch.set_num_threads(4)
    require(str(torch.__version__) == '2.7.1+cu128', 'Checkpoint load runtime differs')
    observations = []; loads = 0
    for name in ('best.pt', 'latest.pt'):
        target = checked(reg['checkpoint_bindings'][name])
        # Project-created exact-SHA new artifacts, loaded only as CPU containers.
        # weights_only rejects unregistered Python pickle globals.
        cp = torch.load(target, map_location='cpu', weights_only=True); loads += 1
        observations.append(inspect(cp, name, acceptance, history, training_registration, torch, slot))
        del cp
        checked(reg['checkpoint_bindings'][name])
    require(loads == 2, 'Bounded CPU load count differs')
    result = {'version': 'TCN-ED6-bounded-checkpoint-audit-1.0', 'passed': True,
              'slot_id': slot,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started,
              'worker_pid': os.getpid(), 'registration_binding': identity(path),
              'source_binding': identity(Path(__file__)), 'observations': observations,
              'actual_new_checkpoint_CPU_loads': loads, 'models_constructed': 0,
              'optimizers_constructed': 0, 'forwards': 0, 'backwards': 0, 'optimizer_updates': 0,
              'historical_checkpoint_loads': 0, 'E_record_reads': 0, 'RNG_restored': False,
              'new_market_evaluation': False, 'claim': 'New CPU container integrity only; no inference or real GPU continuation.'}
    with (trial / 'checkpoint_validation.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'slot_id': slot, 'actual_CPU_loads': loads,
                     'output': identity(trial / 'checkpoint_validation.json')},
                     ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
