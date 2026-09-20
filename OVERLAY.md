# Ramp Kit overlay (customer convention-as-code)

This checkout is a **fork of huggingface/diffusers**. The overlay lives in a
separate repo: [alex-16moro/diffuser_agent](https://github.com/alex-16moro/diffuser_agent).

Cloud Agent install clones it to `ramp-kit/` (gitignored). Do not edit
upstream `AGENTS.md` / `.ai/` — those stay Hugging Face's.

PRs from this overlay are **fork-demo only, not for upstream**. Keep them draft
and titled `[fork demo — not for upstream]`. Overlay clearance is the customer
gate (`ramp-kit/tools/convention_check.py` on the new file). Upstream GitHub
Actions may go red; that is expected — we do not claim Hugging Face's CI.

## Grounding (default path — no MCP)

1. Read `src/diffusers/schedulers/scheduling_euler_discrete.py` and
   `scheduling_ddpm.py` (code beats the philosophy doc).
2. Run the gate. `ramp-kit/conventions/rules.yaml` is authoritative.
3. Optional CLI docs query (same server as MCP, no OAuth):

```bash
python3 ramp-kit/tools/docs_mcp_server.py --query "scheduler set_timesteps step SchedulerMixin register_to_config"
```

`.cursor/mcp.json` is **empty by default** so Cloud launches do not hit Hub
OAuth or stdio cwd failures. Opt-in servers: `.cursor/mcp.optional.json`.

## Demo (Cloud Agent)

1. On the **kit** first: `make demo-maintain` — one YAML edit, five surfaces.
2. Then launch on **this fork** (`alex-16moro/diffusers`), overlay branch.
3. Scaffold writes `src/diffusers/schedulers/scheduling_euler_lite.py` here.
   Gate: `python3 ramp-kit/tools/convention_check.py <that file>` (never `--all`).
4. Open the PR **on this fork**, not on huggingface/diffusers.

Catch-early fixture: `ramp-kit/examples/candidate_scheduler` — do not copy it.
