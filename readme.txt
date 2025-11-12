# Analytics Coordinator MCP Server

A Model Context Protocol (MCP) server that orchestrates Supabase queries and transforms datasets into visualization payloads for other MCP servers such as QuickChart and MPLSoccer.

## Purpose

This MCP server provides a bridge layer so AI assistants can fetch data, reshape it into chart or pitch primitives, and optionally render images without writing glue code each time.

## Features

### Current Implementation

- **`prepare_visual_payload`** - Converts tabular datasets into QuickChart configs or MPLSoccer primitives.
- **`run_pipeline`** - Executes high-level pipelines (SQL → transform → optional render) and returns either render instructions or base64 images.

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp`)
- For Supabase integration: `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (configure via secrets/env)
- For QuickChart rendering: optional `QUICKCHART_API_KEY`

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

- "Convert these Supabase rows into QuickChart line chart config."
- "Run a pipeline that queries Supabase, groups shots, and prepares mplsoccer primitives."
- "Fetch player data, build a bar chart, and render it via QuickChart."

## Architecture

Claude Desktop → MCP Gateway → Analytics Coordinator MCP Server → Supabase / QuickChart / Visualization MCPs

## Development

### Local Testing

```bash
# Configure environment for testing
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="service-role-key"
export QUICKCHART_API_KEY="quickchart-key"

# Run directly
python analytics_coordinator_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python analytics_coordinator_server.py
```

### Adding New Tools

1. Add the function to `analytics_coordinator_server.py`
2. Decorate with `@mcp.tool()`
3. Update documentation and catalog entry
4. Rebuild the Docker image

### Troubleshooting

**Missing Data**
- Ensure Supabase SQL returns `columns` and `rows`
- Validate mapping keys exist in dataset

**Rendering Errors**
- Confirm QuickChart or downstream MCP is accessible
- Check environment variables for required keys

### Security Considerations

- Supabase and QuickChart keys read from environment (use Docker secrets)
- No sensitive data logged
- Runs as non-root user inside container

### License

MIT License
