# /search-docs — library docs in THIS checkout (optional)

Usage: `/search-docs <query>`

Prefer source + gate first (MCP is opt-in; `.cursor/mcp.json` is empty):

- `src/diffusers/schedulers/scheduling_euler_discrete.py`
- `src/diffusers/schedulers/scheduling_ddpm.py`
- `ramp-kit/conventions/rules.yaml`

Then, optional CLI (same server as MCP, no OAuth):

```bash
python3 ramp-kit/tools/docs_mcp_server.py --query $1
```

If `$1` is empty:

```bash
python3 ramp-kit/tools/docs_mcp_server.py --query "scheduler set_timesteps step SchedulerMixin register_to_config"
```

Provenance should say `diffusers checkout`, not `bundled snapshot`. If the
MCP tool `search_docs` is in your list, call that instead.
