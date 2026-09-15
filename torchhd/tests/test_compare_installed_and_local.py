import os
import sys
import subprocess
import tempfile
import json
import textwrap
import pytest


RUNNER = textwrap.dedent(
    """
import time
import torch
import torchhd

torch.manual_seed(0)

# Build two reproducible MAP tensors of 1000 elements
vec_len = 1000
seed = 0
torch.manual_seed(seed)
raw_a = (torch.rand(vec_len) > 0.5).to(dtype=torch.int8)
torch.manual_seed(seed + 1)
raw_b = (torch.rand(vec_len) > 0.5).to(dtype=torch.int8)

# Wrap as MAPTensor if available
try:
    A = torchhd.MAPTensor(raw_a)
    B = torchhd.MAPTensor(raw_b)
except Exception:
    A = raw_a
    B = raw_b

# Warm-up
for _ in range(5):
    _ = A.bind(B)

repeats = 200
start = time.perf_counter()
for _ in range(repeats):
    out = A.bind(B)
end = time.perf_counter()

out_list = out.to(torch.int8).cpu().numpy().tolist()
result = {"out": out_list, "time": end - start}
print(json.dumps(result))
"""
)


def run_runner(python_exe, extra_env):
    fd, path = tempfile.mkstemp(suffix="_runner.py")
    os.close(fd)
    with open(path, "w") as f:
        f.write(RUNNER)

    env = os.environ.copy()
    env.update(extra_env or {})

    proc = subprocess.run([python_exe, path], env=env, capture_output=True, text=True)
    os.remove(path)
    if proc.returncode != 0:
        raise RuntimeError(f"Runner failed: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip())


@pytest.mark.skipif(sys.executable is None, reason="No Python executable")
def test_compare_installed_and_local():
    python_exe = sys.executable

    # 1) Try to run using the installed torchhd (avoid local cwd)
    env_installed = {"PYTHONPATH": ""}
    try:
        installed = run_runner(python_exe, env_installed)
    except Exception as e:
        pytest.skip(f"Could not run installed torchhd: {e}")

    # 2) Run using the local checkout by forcing PYTHONPATH to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    env_local = {"PYTHONPATH": repo_root}
    local = run_runner(python_exe, env_local)

    # Compare outputs
    assert installed["out"] == local["out"], "Outputs differ between installed and local torchhd builds"

    # Print timing info for user inspection and an assertion that local is not slower by an unreasonable factor
    installed_time = installed["time"]
    local_time = local["time"]

    # Allow local to be slower; alert if it's more than 10x slower
    assert local_time / max(installed_time, 1e-12) < 10.0, f"Local build is too slow: {local_time} vs installed {installed_time}"

    print(f"installed_time={installed_time:.6f}s local_time={local_time:.6f}s speedup={installed_time/local_time:.3f}")
