"""Portable H50 cache preparation, admission and cold-start training."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

from shared.io import atomic_json, sha
from shared.paths import ROOT, resolve_path
from software.training.tcn_research_data_p2 import validate_prepared
from software.training.tcn_existing_data_ed4_features import CachedRelativeWindows, prepare, validate_cache
from software.training.tcn_existing_data_ed4_train import SETTINGS, exact_rank, stable_ce
from software.training.tcn_research_evaluate_p2 import metrics, state_digest


def metadata(path, expected):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected:
        raise ValueError('Metadata hash differs. ' + str(path))

    def relocate(value):
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in {'path', 'records_path', 'label_path'} and isinstance(item, str):
                item = resolve_path(item) if '://' in item else Path(item)
                item = item if item.is_absolute() else path.parent/item
                result[key] = str(item.resolve())
            else:
                result[key] = relocate(item)
        return result

    result = relocate(json.loads(path.read_text('utf-8-sig')))
    result.update(_manifest_path=str(path), _manifest_sha256=expected)
    return result


def inputs(data_path, data_sha, cache_path, cache_sha):
    data = validate_prepared(metadata(data_path, data_sha))
    cache = metadata(cache_path, cache_sha)
    roots = dict(cache_root=Path(cache_path).resolve().parent, data_root=Path(data_path).resolve().parent)
    validate_cache(data, cache, **roots)
    if cache.get('p2_manifest_binding', {}).get('sha256') != data_sha:
        raise ValueError('Cache parent manifest differs')
    verify_payloads(data, cache)
    return data, cache, roots


def verify_payloads(data, cache):
    for day, ref in cache['cache_files'].items():
        path = Path(ref['path'])
        label = data['day_label_reports'][day]
        if path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']:
            raise ValueError('Normalized cache changed. ' + day)
        path = Path(label['label_path'])
        if path.stat().st_size != label['label_bytes'] or sha(path) != label['label_sha256']:
            raise ValueError('Label stream changed. ' + day)


def epoch(model, dataset, epoch_number, device, optimizer, np, torch):
    training = optimizer is not None
    model.train(training)
    order = np.random.default_rng(42 + epoch_number).permutation(len(dataset)) if training else np.arange(len(dataset))
    confusion = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.0
    updates = 0
    with torch.set_grad_enabled(training):
        for begin in range(0, len(order), 256):
            x, y = dataset.batch(order[begin:begin+256])
            if not np.isfinite(x).all() or not np.all((y >= 0) & (y <= 2)):
                raise ValueError('Invalid training tensor or class')
            tx = torch.from_numpy(x).to(device)
            ty = torch.from_numpy(y).to(device)
            logits = model(tx)
            loss = torch.nn.functional.cross_entropy(logits, ty)
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise ValueError('Nonfinite training logits or loss')
            z = logits.detach().cpu().numpy()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise ValueError('Missing or nonfinite gradient')
                optimizer.step()
                updates += 1
                loss_sum += float(loss.detach().cpu()) * len(y)
            else:
                loss_sum += float(stable_ce(z, y, np).sum(dtype=np.float64))
            confusion += np.bincount(3*y + z.argmax(1), minlength=9).reshape(3, 3)
    result = metrics(confusion.tolist())
    if result['samples'] != len(dataset):
        raise ValueError('Epoch window accounting differs')
    return dict(result, loss=loss_sum/len(dataset), optimizer_updates=updates)


def train(data, cache, roots, output, device, epochs, identity):
    import numpy as np
    import torch
    from software.model.tcn_research_p5_model import build_model

    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    datasets = []
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', device=device, data_kind=data.get('data_kind', 'historical'),
                  input_identity=identity, training_settings=dict(SETTINGS, device=device, max_epochs=epochs),
                  normalization=cache['normalization'], software=dict(torch=str(torch.__version__), numpy=np.__version__),
                  source_sha256={name: sha(ROOT/name) for name in (
                      'software/training/entry.py', 'software/training/tcn_existing_data_ed4_features.py',
                      'software/model/tcn_research_p5_model.py', 'software/model/tcn_candidate.py')}, history=[])
    atomic_json(output/'result.json', result)
    try:
        datasets = [CachedRelativeWindows(data, cache, role, **roots) for role in ('train', 'tuning_validation')]
        model = build_model(weight_norm=True).to(device)
        result['initial_state_sha256'] = state_digest(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        best = None
        wait = 0
        for number in range(1, epochs+1):
            training = epoch(model, datasets[0], number, device, optimizer, np, torch)
            development = epoch(model, datasets[1], number, device, None, np, torch)
            rank = exact_rank(development['confusion_true_rows_predicted_columns'])
            improved = best is None or rank > best
            wait = 0 if improved else wait+1
            row = dict(epoch=number, train=training, development=development, improved=improved, wait=wait)
            result['history'].append(row)
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=number,
                              config=result['training_settings'], history=result['history'],
                              normalization=cache['normalization'], input_identity=identity,
                              state_sha256=state_digest(model), data_kind=result['data_kind'])
            temporary = output/'checkpoint.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(output/('epoch'+str(number).zfill(2)+'.pt'))
            if improved:
                best = rank
                result['selected_epoch'] = number
                torch.save(checkpoint, temporary)
                temporary.replace(output/'best.pt')
            atomic_json(output/'result.json', result)
            if wait >= 3:
                break
        verify_payloads(data, cache)
        result.update(status='completed', selected_checkpoint_sha256=sha(output/'best.pt'),
                      stop_reason='patience3' if wait >= 3 else 'epoch_limit',
                      optimizer_updates=sum(row['train']['optimizer_updates'] for row in result['history']))
        atomic_json(output/'normalization.json', cache['normalization'])
        return result
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + '. ' + str(exc))
        raise
    finally:
        for dataset in datasets:
            dataset.close()
        atomic_json(output/'result.json', result)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(argument.split('=', 1)[0] == '--registration' for argument in argv):
        from software.training.tcn_existing_data_ed4_train import main as historical_main
        return historical_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'admit', 'run'])
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--data-sha256', required=True)
    parser.add_argument('--cache-manifest', type=Path)
    parser.add_argument('--cache-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, choices=range(1, 11), default=10)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if args.output is None:
            parser.error('prepare requires --output')
        data = validate_prepared(metadata(args.data_manifest, args.data_sha256))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = prepare(data, args.output, cache_root=args.output.resolve().parent)
        print(json.dumps(dict(status='prepared', manifest=str(args.output/'cache_manifest.json'),
                              sha256=sha(args.output/'cache_manifest.json'), samples=result['sample_counts'])))
        return 0
    if args.cache_manifest is None or args.cache_sha256 is None:
        parser.error('admit and run require --cache-manifest and --cache-sha256')
    data, cache, roots = inputs(args.data_manifest, args.data_sha256, args.cache_manifest, args.cache_sha256)
    identity = dict(data_manifest_sha256=args.data_sha256, cache_manifest_sha256=args.cache_sha256)
    if args.action == 'admit':
        print(json.dumps(dict(status='admitted', sample_counts=data['sample_counts'], input_identity=identity)))
        return 0
    if args.output is None:
        parser.error('run requires --output')
    result = train(data, cache, roots, args.output, args.device, args.epochs, identity)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
