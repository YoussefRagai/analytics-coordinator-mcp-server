# Analytics Coordinator MCP Server Notes

## Overview
- **Service name:** Analytics Coordinator
- **Entry script:** `analytics_coordinator_server.py`
- **Purpose:** Orchestrate Supabase queries and convert datasets into visualization payloads or rendered assets.

## Configuration
- Env vars:
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (optional; required for SQL pipelines)
  - `SUPABASE_TIMEOUT_SECONDS` (default `20`)
  - `QUICKCHART_BASE_URL` (default `https://quickchart.io`)
  - `QUICKCHART_API_KEY` (optional)
  - `QUICKCHART_TIMEOUT_SECONDS` (default `20`)
  - `COORDINATOR_INLINE_LIMIT` (default `6000`)
- No filesystem output; responses embed JSON payloads or base64 images.

## Tools
- `prepare_visual_payload(dataset="", mapping="", target="")`
  - Accepts dataset JSON (`columns`, `rows`) or row list, mapping spec:
    - `target` = `mplsoccer`: mapping must include `x`, `y`, optional `group_by`, `marker_by`, etc.
    - `target` = `quickchart`: mapping provides `x`, `y`, optional `series`, `type`, `options`, etc.
  - Returns normalized payload for downstream MCP.
- `run_pipeline(plan="")`
  - Executes optional Supabase SQL, transforms dataset, and optionally renders.
  - Plan JSON keys: `sql`, `dataset`, `target`, `mapping`, `render`, `title`, `save_as`.
  - When `render=true` and `target="mplsoccer"`, returns a `tools/call` payload to send to `draw_pitch`.
  - When `render=true` and `target="quickchart"`, returns base64-encoded chart image.

## Error Handling
- Validates mapping/dataset compatibility.
- Gracefully reports Supabase/QuickChart HTTP failures.
- Ensures informative ❌ messages when required fields are missing.

## Logging
- INFO-level logging to stderr via `logging.basicConfig`.
- Logs external API errors with status codes.

## Docker Runtime
- Based on `python:3.11-slim`.
- Dependencies: `mcp[cli]`, `httpx`.
- Runs as non-root `mcpuser`.

## Testing Tips
```bash
export SUPABASE_URL="https://example.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="service-role-key"
python analytics_coordinator_server.py
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python analytics_coordinator_server.py
```

## Maintenance
- Keep tool docstrings single-line.
- When adding adapters, reuse helper functions (`_tabular_to_chartjs`, `_mpl_primitives_from_dataset`).
- Consider expanding `run_pipeline` for additional targets (e.g., wordcloud) as other MCPs are added.
