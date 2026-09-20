---
name: search-docs
description: Optional docs CLI for this diffusers checkout. Prefer scheduler source + the gate; MCP is opt-in.
---

# Search this library's docs

```bash
python3 ramp-kit/tools/docs_mcp_server.py --query "<the question>"
```

Cite provenance (`diffusers checkout` vs bundled snapshot). Prefer reading
`src/diffusers/schedulers/scheduling_euler_discrete.py` and
`scheduling_ddpm.py` first. `ramp-kit/conventions/rules.yaml` is the gate.
`.cursor/mcp.json` is empty by default.
