"""Run migrated runtime contracts on Linux without importing Torch or using GPUs."""
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
    'revoked_ready_check', 'decode_run_check', 'kv_fraction_check', 'trace_check',
)


def main():
    if sys.platform != 'linux' or sys.version_info < (3, 10):
        raise SystemExit('Use Linux Python 3.10+ (WSL is sufficient) for AF_UNIX/pidfd checks.')
    commands = [[str(ROOT/'acceptance'/(name+'.py'))] for name in CHECKS]
    commands += [[str(ROOT/'acceptance/runtime_check.py'), '--source', '--gate'],
                 [str(ROOT/'acceptance/native_channel_close_check.py'), '--require-zmq'],
                 ['-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-p', 'test_*.py']]
    for args in commands:
        print('CPU_CHECK '+' '.join(args), flush=True)
        subprocess.run([sys.executable, '-I', '-B', *args], cwd=ROOT, check=True, timeout=180)
    print('CPU_SUITE_PASS; no GPU correctness or performance claim', flush=True)


if __name__ == '__main__':
    main()
