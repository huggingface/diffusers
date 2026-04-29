import argparse
import json
import os
import re
import urllib.request
import urllib.error


def find_existing_issues(repo: str, token: str, title_prefix: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/issues?state=open&labels=ci-failure"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "diffusers-ci-bot")

    try:
        with urllib.request.urlopen(req) as resp:
            issues = json.loads(resp.read())
        return [i for i in issues if title_prefix in i.get("title", "")]
    except Exception:
        return []


def create_issue(repo: str, token: str, title: str, body: str, label: str = "ci-failure") -> str:
    url = f"https://api.github.com/repos/{repo}/issues"
    data = json.dumps({"title": title, "body": body, "labels": [label]}).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "diffusers-ci-bot")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result.get("html_url", "")


def classify_error(error_text: str) -> str:
    if "out of memory" in error_text.lower() or "OOM" in error_text:
        return ("**错误类型**: OOM (显存不足)\n\n"
                "**建议**: 减小 `num_frames`、`width`/`height`，或启用 `enable_attention_slicing`")
    if "connection" in error_text.lower() or "timeout" in error_text.lower():
        return ("**错误类型**: 网络/连接异常\n\n"
                "**建议**: 检查 HuggingFace/ModelScope 网络连通性，或切换到本地权重")
    if "key" in error_text.lower() or "AttributeError" in error_text:
        return ("**错误类型**: API 不兼容\n\n"
                "**建议**: 检查 diffusers 版本与模型是否匹配，检查参数名是否正确（true_cfg_scale vs guidance_scale）")
    if "import" in error_text.lower() or "ModuleNotFoundError" in error_text.lower():
        return ("**错误类型**: 依赖缺失\n\n"
                "**建议**: 检查 requirements，安装缺失的依赖包")
    return ("**错误类型**: 未知\n\n"
            "**建议**: 请查看上方错误日志进行人工排查")


def build_issue_title(pipeline: str, variant: str, date_str: str) -> str:
    return f"[CI] {pipeline} / {variant} 运行失败 ({date_str})"


def build_issue_body(r: dict, run_url: str) -> str:
    error_text = r.get("error", "无错误信息")
    lines = [
        "## 失败信息",
        f"- **Pipeline**: {r['pipeline']}",
        f"- **Variant**: {r['variant']}",
        f"- **配置**: {r['config_name']}",
        f"- **设备**: {r['device']} / {r['dtype']}",
        f"- **并行策略**: {r['parallel']}",
        "",
        "## 修复建议",
        classify_error(error_text),
        "",
        "## 错误日志",
        "```",
        error_text[:5000],
        "```",
        "",
        f"## 完整日志",
        f"[Actions Run]({run_url})" if run_url else run_url,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="output directory containing all_results.json")
    args = parser.parse_args()

    repo = os.environ.get("GITHUB_REPOSITORY", "luren55/diffusers")
    token = os.environ.get("GITHUB_TOKEN", "")
    run_url = ""
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if server_url and repo and run_id:
        run_url = f"{server_url}/{repo}/actions/runs/{run_id}"

    results_path = os.path.join(args.report, "all_results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: {results_path} not found")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    failed = [r for r in results if r["result"] and r["result"].get("status") == "failed"]

    if not failed:
        print("No failures, no issues to create.")
        return

    date_str = r["timestamp"][:10] if failed else "unknown-date"

    for r in failed:
        title = build_issue_title(r["pipeline"], r["variant"], date_str)
        title_prefix = f"[CI] {r['pipeline']} / {r['variant']}"

        existing = find_existing_issues(repo, token, title_prefix)
        if existing:
            print(f"[SKIP] Issue already exists for {r['pipeline']} / {r['variant']}: {existing[0].get('html_url')}")
            continue

        body = build_issue_body(r, run_url)
        try:
            url = create_issue(repo, token, title, body)
            print(f"[OK] Created issue: {url}")
        except Exception as e:
            print(f"[FAIL] Could not create issue: {e}")


if __name__ == "__main__":
    main()
