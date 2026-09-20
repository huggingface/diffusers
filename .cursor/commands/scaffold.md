# /scaffold — first contribution on the REAL library checkout

Usage: `/scaffold <component> <Name>`

This workspace is `huggingface/diffusers` (fork). Overlay kit is `ramp-kit/`
(cloned from alex-16moro/diffuser_agent). Do not overwrite `.ai/` or root `AGENTS.md`.

`$1` = component (`scheduler`). `$2` = PascalCase name without suffix (`EulerLite`
→ `EulerLiteScheduler`, `scheduling_euler_lite.py`).

## 0. Ground first

Prefer **this checkout's source** over memory and over MCP:

- `src/diffusers/schedulers/scheduling_euler_discrete.py`
- `src/diffusers/schedulers/scheduling_ddpm.py`

Those files are the contract. `ramp-kit/conventions/rules.yaml` / the gate is
authoritative if anything disagrees (the philosophy doc is stale).

Optional docs CLI (MCP is opt-in; `.cursor/mcp.json` is empty by default):

```bash
python3 ramp-kit/tools/docs_mcp_server.py --query "scheduler set_timesteps step SchedulerMixin register_to_config"
```

Cite provenance if you run that. Do not read or copy
`ramp-kit/examples/candidate_scheduler/` into the new files.

## 1. Paths (library layout, not the kit stand-in)

- Implementation: `src/diffusers/schedulers/scheduling_euler_lite.py`
- Test: `tests/schedulers/test_scheduling_euler_lite.py`

## 2. Copy from the overlay templates

- `ramp-kit/templates/scheduler/scheduling_TEMPLATE.py` → implementation
- Rename `TemplateScheduler` → `$2Scheduler`
- Leave `TODO(engineer)` in `step`. Do not invent Euler math.

## 3. TEST001

Copy `ramp-kit/tests/_templates/scheduler_test.py`. Set `TARGET` and `CLASS`.
The test must mention `set_timesteps` and `step`, and (TEST002) include
assertions, same-seed determinism, and shape/dtype checks.

## 4. Gate (file-scoped — do not `--all` this library)

```bash
python3 ramp-kit/tools/convention_check.py ramp-kit/examples/candidate_scheduler
python3 ramp-kit/tools/convention_check.py src/diffusers/schedulers/scheduling_euler_lite.py
python3 -m unittest tests.schedulers.test_scheduling_euler_lite -v
```

Catch-early uses the kit fixture path. Stop at 0 blocking. Do not fabricate numerics.
Do not open a PR against huggingface/diffusers — PR this fork.
