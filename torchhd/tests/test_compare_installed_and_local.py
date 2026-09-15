import os
import sys
import subprocess
import tempfile
import json
import textwrap
import pytest
import json


RUNNER = textwrap.dedent(
    """
import time
import json
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
def test_compare_installed_distributions():
    """Locate two installed distributions (torchHDrvv and torch-hd) and compare their inference outputs and timings.

    This test will look for the distributions by name and, if found, run the same runner
    with `PYTHONPATH` pointed to each distribution's package directory so both can be
    imported independently in separate subprocesses.
    """
    python_exe = sys.executable

    # Try to locate both distributions using importlib.metadata
    try:
        from importlib import metadata
    except Exception:
        import importlib_metadata as metadata

    candidates = ["torchHDrvv", "torch-hd", "torch_hd", "torchhd"]
    found = []
    for name in candidates:
        try:
            dist = metadata.distribution(name)
            base = str(dist.locate_file(""))
            pkg_dir = os.path.join(base, "torchhd")
            if os.path.isdir(pkg_dir):
                found.append((name, base))
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    if len(found) < 2:
        pytest.skip("Need two installed distributions (torchHDrvv and torch-hd) present in the environment to compare")

    # Use first two found distributions
    (name_a, base_a), (name_b, base_b) = found[0], found[1]

    a = run_runner(python_exe, {"PYTHONPATH": base_a})
    b = run_runner(python_exe, {"PYTHONPATH": base_b})

    assert a["out"] == b["out"], f"Outputs differ between {name_a} and {name_b}"

    time_a = a["time"]
    time_b = b["time"]

    # sanity check: neither should be more than 10x slower
    assert time_a / max(time_b, 1e-12) < 10.0 and time_b / max(time_a, 1e-12) < 10.0

    print(f"{name_a}: {time_a:.6f}s  {name_b}: {time_b:.6f}s  speedup={time_b/time_a:.3f}")
