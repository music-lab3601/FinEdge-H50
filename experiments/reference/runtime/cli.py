import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='FinEdge ED4 independent host baseline')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('admit', 'verify-cpu', 'export', 'compile', 'verify-npu', 'verify-host', 'board-check', 'board-install', 'board-start', 'gui', 'extract-itch', 'rebuild-itch'):
        sub.add_parser(name)
    run = sub.add_parser('run')
    run.add_argument('--mode', choices=('cpu', 'onnx', 'host', 'socket', 'socket-verify'), default='socket')
    run.add_argument('--rounds', type=int, default=1)
    run.add_argument('--duration', type=float, default=0)
    run.add_argument('--rate', type=float, default=0)
    run.add_argument('--dataset', choices=('delivery1024','nasdaq20190730','itch20190730','itch20190730_gzip'), default='delivery1024')
    run.add_argument('--queue-capacity', type=int, default=8, help='raw event producer queue; lossless file backpressure')
    run.add_argument('--max-windows', type=int, default=0, help='0 means entire dataset')
    run.add_argument('--loop', action='store_true')
    args = parser.parse_args(argv)
    if args.command in ('extract-itch','rebuild-itch'):
        from experiments.reference.runtime.itch import extract_aapl, rebuild
        result=(extract_aapl if args.command=='extract-itch' else rebuild)()
        result={k:v for k,v in result.items() if k not in ('segments','control_events')}
    elif args.command == 'admit':
        from software.runtime.delivery import admit_deliveries
        result = admit_deliveries()
    elif args.command == 'verify-cpu':
        from experiments.reference.runtime.verification import verify_cpu
        result = verify_cpu()
    elif args.command == 'export':
        from deployment.export.onnx import export_onnx
        result = export_onnx()
    elif args.command == 'compile':
        from deployment.export.onnx import compile_icraft
        result = compile_icraft()
    elif args.command in ('verify-npu', 'verify-host'):
        from experiments.reference.runtime.verification import verify_icraft
        result = verify_icraft(args.command == 'verify-npu')
        result = {'status': result['status'], 'summary': result['summary']}
    elif args.command == 'run':
        from experiments.reference.runtime.replay import ReplayControl, run_replay
        if args.rounds < 1 or args.duration < 0 or args.rate < 0 or args.max_windows < 0 or args.queue_capacity < 1:
            parser.error('rounds >=1, duration/rate >=0 required')
        def emit(event):
            if event['type'] in ('ready', 'round', 'error', 'finished'):
                print(json.dumps(event, ensure_ascii=False), flush=True)
        result = run_replay(args.mode, args.rounds, args.duration, ReplayControl(args.rate), emit,
                            dataset=args.dataset, max_windows=args.max_windows,repeat=args.loop,
                            queue_capacity=args.queue_capacity)
    elif args.command.startswith('board-'):
        from experiments.reference.runtime.board import install_runtime, preflight, start_service
        result = {'board-check': preflight, 'board-install': install_runtime, 'board-start': start_service}[args.command]()
    else:
        from experiments.reference.runtime.gui import main as gui
        return gui()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if isinstance(result, dict) and result.get('status') == 'failed' else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)

