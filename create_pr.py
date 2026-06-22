import urllib.request, json, os, ssl, subprocess
ctx = ssl.create_default_context(); ctx.check_hostname = True; ctx.verify_mode = ssl.CERT_REQUIRED
token = os.environ.get("GITHUB_TOKEN", "")
try:
    subprocess.run(["git", "config", "user.email", "kartavyaniraj.dikshit2021@vitstudent.ac.in"], check=False)
    subprocess.run(["git", "config", "user.name", "KartavyaDikshit"], check=False)
    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "--signoff", "-m", "fix: ensure custom timesteps are consistent with sigmas in FlowMatchEulerDiscreteScheduler"], check=False)
    subprocess.run(["git", "push", f"https://{token}@github.com/KartavyaDikshit/diffusers.git", "fix-issue-14013", "--force"], check=False)
except: pass
payload = {"title": "fix: ensure custom timesteps are consistent with sigmas in FlowMatchEulerDiscreteScheduler", "body": "Resolves #14013\n\nWhen custom timesteps were provided to set_timesteps with shift != 1, the timesteps array would diverge from the sigmas array after internal computation. This fix ensures custom timesteps are processed consistently with the shift factor, matching the behavior of auto-generated timesteps.\n\nSigned-off-by: KartavyaDikshit <kartavyaniraj.dikshit2021@vitstudent.ac.in>", "head": "KartavyaDikshit:fix-issue-14013", "base": "main"}
req = urllib.request.Request("https://api.github.com/repos/huggingface/diffusers/pulls", data=json.dumps(payload).encode(), headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        pr_data = json.loads(r.read())
        print("[+] PR_CREATED:", pr_data["number"])
except Exception as e: print("[!] PR Failed:", e)
