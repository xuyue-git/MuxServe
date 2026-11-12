"""Run migrated runtime contracts on Linux without importing Torch or using GPUs.

    python -I -B tools/check_cpu.py [--require-source]

Three checks read the pinned SGLang source under third_party/ (materialized by
tools/bootstrap_source.py). Without it they are reported as skipped and the
suite ends with CPU_SUITE_PASS_WITH_SKIPS; --require-source makes a missing
source an error instead.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    'storage_check', 'controller_check', 'deployment_check', 'driver_history_check',
    'controller_projection_check', 'block_validation_check', 'metadata_upload_check',
    'command_snapshot_check', 'command_retention_check', 'control_plane_regression_check',
    'rpc_framing_check', 'rpc_admission_check', 'ipc_teardown_check', 'watchdog_teardown_check',
    'shutdown_check', 'mps_exit_check', 'startup_smoke_check', 'startup_budget_check',
    'startup_regression_check', 'weight_owner_collectives_check', 'idle_cuda_cache_check',
    'revoked_ready_check', 'decode_run_check', 'kv_fraction_check', 'trace_check', 'decode_graph_check',
    'fast_path_check', 'owner_speed_check', 'no_vllm_check',
)
# Checks that read third_party/sglang/python (the pinned, patched SGLang source).
NEEDS_SOURCE = ('watchdog_teardown_check', 'startup_regression_check', 'runtime_check')
SOURCE = ROOT / 'third_party' / 'sglang' / 'python' / 'sglang'


def main():
    if sys.platform != 'linux' or sys.version_info < (3, 10):
        raise SystemExit('Use Linux Python 3.10+ (WSL is sufficient) for AF_UNIX/pidfd checks.')
    require_source = '--require-source' in sys.argv[1:]
    have_source = SOURCE.is_dir()
    if not have_source and require_source:
        raise SystemExit(f'Pinned SGLang source missing at {SOURCE}; run python tools/bootstrap_source.py')
    commands = [[str(ROOT/'acceptance'/(name+'.py'))] for name in CHECKS]
    commands += [[str(ROOT/'acceptance/runtime_check.py'), '--source', '--gate'],
                 [str(ROOT/'acceptance/native_channel_close_check.py'), '--require-zmq'],
                 ['-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-p', 'test_*.py']]
    skipped = []
    for args in commands:
        name = Path(args[0]).stem
        if name in NEEDS_SOURCE and not have_source:
            print(f'CPU_CHECK_SKIPPED {name}: needs the pinned SGLang source at {SOURCE} '
                  '(python tools/bootstrap_source.py)', flush=True)
            skipped.append(name)
            continue
        print('CPU_CHECK '+' '.join(args), flush=True)
        subprocess.run([sys.executable, '-I', '-B', *args], cwd=ROOT, check=True, timeout=180)
    if skipped:
        print(f'CPU_SUITE_PASS_WITH_SKIPS skipped={skipped}; no GPU correctness or performance claim',
              flush=True)
    else:
        print('CPU_SUITE_PASS; no GPU correctness or performance claim', flush=True)


if __name__ == '__main__':
    main()
