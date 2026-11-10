import os
import signal
import sys

from huggingface_hub import HfApi


COMMAND = r"""
set -euo pipefail
git init -q /tmp/diffusers
cd /tmp/diffusers
git remote add origin "https://github.com/${REPOSITORY}.git"
git fetch -q --depth=2 origin "refs/pull/${PR_NUMBER}/head"
git checkout -q --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$PR_SHA"
nvidia-smi
printf 'torch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n' > "$UV_OVERRIDE"
uv pip install -e ".[quality,training,test]"
uv pip install peft@git+https://github.com/huggingface/peft.git
uv pip uninstall accelerate && uv pip install -U accelerate@git+https://github.com/huggingface/accelerate.git
uv pip uninstall transformers huggingface_hub && UV_PRERELEASE=allow uv pip install -U transformers@git+https://github.com/huggingface/transformers.git
diffusers-cli env
status=0
python - <<'PY' || status=$?
import os
import shlex
import subprocess

args = shlex.split(os.environ["PYTEST_ARGS"])
raise SystemExit(subprocess.call(["pytest", "--make-reports=tests_bot_gpu", *args]))
PY
if [[ "$status" -ne 0 ]]; then
    cat reports/tests_bot_gpu_stats.txt || true
    cat reports/tests_bot_gpu_failures_short.txt || true
fi
exit "$status"
"""


def main():
    flavor = os.environ["PYTEST_FLAVOR"]
    namespace = os.environ["HF_JOBS_NAMESPACE"]
    token = os.environ["HF_TOKEN"]
    if not namespace:
        raise SystemExit("HF_JOBS_NAMESPACE must be set before running HF Jobs")
    if not token:
        raise SystemExit("Set the DIFFUSERS_HF_JOBS_TOKEN repository secret before running HF Jobs")

    api = HfApi(token=token)
    flavors = {
        hardware.name
        for hardware in api.list_jobs_hardware()
        if hardware.accelerator is not None
        and hardware.accelerator.type.lower() == "gpu"
        and int(hardware.accelerator.quantity) > 1
    }
    if flavor not in flavors:
        raise SystemExit(f"Unsupported multi-GPU flavor: {flavor}. Choose from: {', '.join(sorted(flavors))}")

    job = api.run_job(
        image="diffusers/diffusers-pytorch-cuda",
        command=["bash", "-lc", COMMAND],
        flavor=flavor,
        namespace=namespace,
        timeout="2h",
        name=f"diffusers-pr-{os.environ['PR_NUMBER']}-pytest",
        env={
            "REPOSITORY": os.environ["REPOSITORY"],
            "PR_NUMBER": os.environ["PR_NUMBER"],
            "PR_SHA": os.environ["PR_SHA"],
            "PYTEST_ARGS": os.environ["PYTEST_ARGS"],
            "DIFFUSERS_IS_CI": "yes",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTEST_TIMEOUT": "600",
            "UV_OVERRIDE": "/tmp/uv-overrides.txt",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        },
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"job_url={job.url}\n")
    print(f"HF Job: {job.url}", flush=True)

    def cancel_job(signum, frame):
        api.cancel_job(job_id=job.id, namespace=namespace)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, cancel_job)
    signal.signal(signal.SIGTERM, cancel_job)

    try:
        finished = api.wait_for_job(job_id=job.id, namespace=namespace, poll_interval=20)
    except Exception:
        api.cancel_job(job_id=job.id, namespace=namespace)
        raise
    try:
        for line in api.fetch_job_logs(job_id=job.id, namespace=namespace, tail=200):
            print(line, flush=True)
    except Exception as error:
        print(f"Could not fetch HF Job logs: {error}", flush=True)
    print(f"HF Job status: {finished.status.stage}", flush=True)
    if finished.status.stage != "COMPLETED":
        sys.exit(1)


if __name__ == "__main__":
    main()
