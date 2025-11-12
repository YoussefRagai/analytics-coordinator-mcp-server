#!/usr/bin/env python3
"""Simple Analytics Coordinator MCP Server - Orchestrate data transforms and visual payloads."""

import os
import sys
import json
import logging
import base64
from datetime import datetime, timezone
from collections import defaultdict, OrderedDict

import httpx
from mcp.server.fastmcp import FastMCP


LOG_LEVEL = os.environ.get("MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("analytics_coordinator-server")

mcp = FastMCP("analytics_coordinator", stateless_http=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_TIMEOUT_SECONDS = float(os.environ.get("SUPABASE_TIMEOUT_SECONDS", "20"))

QUICKCHART_BASE_URL = os.environ.get("QUICKCHART_BASE_URL", "https://quickchart.io").rstrip("/")
QUICKCHART_API_KEY = os.environ.get("QUICKCHART_API_KEY", "").strip()
QUICKCHART_TIMEOUT_SECONDS = float(os.environ.get("QUICKCHART_TIMEOUT_SECONDS", "20"))

MPL_PITCH_DEFAULT = {
    "pitch_type": "statsbomb",
    "orientation": "horizontal",
    "pitch_length": 120,
    "pitch_width": 80,
}

MAX_INLINE_CHARS = int(os.environ.get("COORDINATOR_INLINE_LIMIT", "6000"))


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(raw: str, label: str):
    if not raw.strip():
        return False, f"❌ Error: {label} is required."
    try:
        data = json.loads(raw)
        return True, data
    except json.JSONDecodeError as exc:
        return False, f"❌ Error: Invalid {label} JSON ({exc})."


def _format_json_output(obj, summary: str) -> str:
    try:
        text = json.dumps(obj, indent=2)
    except TypeError:
        text = json.dumps(str(obj))
    if len(text) <= MAX_INLINE_CHARS:
        return f"""✅ Success:
- Payload:
{text}

Summary: {summary} | Generated at {_iso_timestamp()}"""
    preview = text[:MAX_INLINE_CHARS]
    return f"""✅ Success:
- Payload preview (truncated to {MAX_INLINE_CHARS} chars):
{preview}
... (truncated)

Summary: {summary} | Generated at {_iso_timestamp()}"""


async def _supabase_query(sql: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False, "❌ Error: Supabase environment (URL or key) not configured for coordinator."
    body = {"query": sql}
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    timeout = httpx.Timeout(SUPABASE_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/pg_execute_sql",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            return True, resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Supabase query failed with %s", exc.response.status_code)
        return False, f"❌ API Error: {exc.response.status_code} executing Supabase SQL."
    except Exception as exc:
        logger.error("Supabase query error: %s", exc)
        return False, f"❌ Error: {str(exc)}"


def _tabular_to_chartjs(table: dict, spec: dict) -> tuple:
    headers = table.get("columns") or []
    rows = table.get("rows") or []
    if not headers or not rows:
        return False, "❌ Error: Supplied dataset is empty."

    x_col = spec.get("x")
    y_cols = spec.get("y")
    series_col = spec.get("series")
    if not x_col or not y_cols:
        return False, "❌ Error: chart spec requires 'x' and 'y' keys."
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    try:
        x_idx = headers.index(x_col)
    except ValueError:
        return False, f"❌ Error: x column '{x_col}' not found."

    series_idx = None
    if series_col:
        try:
            series_idx = headers.index(series_col)
        except ValueError:
            return False, f"❌ Error: series column '{series_col}' not found."

    y_indices = []
    for col in y_cols:
        try:
            y_indices.append(headers.index(col))
        except ValueError:
            return False, f"❌ Error: y column '{col}' not found."

    datasets = OrderedDict()
    for row in rows:
        key = "Series"
        if series_idx is not None:
            key = str(row[series_idx])
        if key not in datasets:
            datasets[key] = [[] for _ in y_indices]
        for idx, y_pos in enumerate(y_indices):
            datasets[key][idx].append(row[y_pos])

    labels = [row[x_idx] for row in rows]

    chart_type = spec.get("type", "bar")
    dataset_entries = []
    palette = spec.get("palette") or [
        "#ff6b6b",
        "#4ecdc4",
        "#1a535c",
        "#ffa36c",
        "#c77dff",
        "#ffd93d",
        "#6bcf99",
        "#b08ea2",
    ]
    color_iter = 0
    for series_name, values in datasets.items():
        for y_idx, column in enumerate(y_cols):
            label = series_name if len(y_cols) == 1 else f"{series_name} - {column}"
            dataset_entries.append(
                {
                    "label": label,
                    "data": values[y_idx],
                    "backgroundColor": palette[color_iter % len(palette)],
                    "borderColor": palette[color_iter % len(palette)],
                    "fill": chart_type in {"line", "radar"} and spec.get("fill", False),
                }
            )
            color_iter += 1

    chart = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": dataset_entries,
        },
        "options": spec.get("options", {}),
    }
    return True, chart


def _encode_image(content: bytes, mime: str, summary: str) -> str:
    if not content:
        return f"⚠️ Warning: No image content generated.\nSummary: {summary} | Generated at {_iso_timestamp()}"
    encoded = base64.b64encode(content).decode("ascii")
    return f"""✅ Success:
- MIME type: {mime}
- Base64 payload:
{encoded}

Summary: {summary} | Generated at {_iso_timestamp()}"""


async def _quickchart_render(chart: dict, fmt: str, width: str, height: str, background: str) -> tuple:
    body = {"chart": chart}
    if fmt:
        body["format"] = fmt
    if width.strip():
        try:
            body["width"] = int(width)
        except ValueError:
            return False, f"❌ Error: Invalid width '{width}'."
    if height.strip():
        try:
            body["height"] = int(height)
        except ValueError:
            return False, f"❌ Error: Invalid height '{height}'."
    if background.strip():
        body["backgroundColor"] = background.strip()

    headers = {"Content-Type": "application/json"}
    if QUICKCHART_API_KEY:
        headers["X-QuickChart-Api-Key"] = QUICKCHART_API_KEY

    timeout = httpx.Timeout(QUICKCHART_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{QUICKCHART_BASE_URL}/chart", headers=headers, json=body)
            response.raise_for_status()
            return True, response.content
    except httpx.HTTPStatusError as exc:
        logger.error("QuickChart render failed: %s", exc.response.status_code)
        return False, f"❌ API Error: {exc.response.status_code} while rendering chart."
    except Exception as exc:
        logger.error("QuickChart render error: %s", exc)
        return False, f"❌ Error: {str(exc)}"


def _mpl_primitives_from_dataset(dataset, mapping: dict) -> dict:
    rows = dataset
    if isinstance(dataset, dict):
        if isinstance(dataset.get("rows"), list):
            rows = dataset["rows"]
        elif isinstance(dataset.get("data"), list):
            rows = dataset["data"]
        else:
            raise ValueError("dataset must be a list of objects or include 'rows'.")

    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must be a non-empty list of objects.")

    grouping = mapping.get("group_by")
    x_field = mapping.get("x")
    y_field = mapping.get("y")
    color_map = mapping.get("color_map") or {}
    color_map_lower = {str(k).lower(): v for k, v in color_map.items()}
    default_color = mapping.get("default_color", "#ff6b6b")
    marker_map = mapping.get("marker_map") or {}
    marker_map_lower = {str(k).lower(): v for k, v in marker_map.items()}
    size_map = mapping.get("size_map") or {}
    size_map_lower = {str(k).lower(): v for k, v in size_map.items()}
    default_marker = mapping.get("default_marker", "o")
    default_size = float(mapping.get("default_size", 120))
    alpha_map = mapping.get("alpha_map") or {}
    alpha_map_lower = {str(k).lower(): v for k, v in alpha_map.items()}
    default_alpha = float(mapping.get("default_alpha", 0.9))

    if not x_field or not y_field:
        raise ValueError("mapping.x and mapping.y are required.")

    split_sides = bool(mapping.get("split_sides"))
    team_field = mapping.get("team_field") or mapping.get("team") or (grouping if grouping and grouping not in {x_field, y_field} else None)
    layout = mapping.get("layout", {})
    pitch_length = float(layout.get("pitch_length", 100))
    left_teams = mapping.get("left_team_values") or mapping.get("left_teams") or []
    right_teams = mapping.get("right_team_values") or mapping.get("right_teams") or []
    default_right_team = None
    color_field = mapping.get("color_by")
    legend_field = mapping.get("legend_field")
    legend_format = mapping.get("legend_format", "{group} ({marker})" if marker_field else "{group}")

    if split_sides and not team_field:
        raise ValueError("mapping.team_field (or group_by) required when split_sides is true.")

    marker_field = mapping.get("marker_by")
    extras_fields = mapping.get("extras", [])

    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("dataset rows must be objects with keys.")
        if x_field not in row or y_field not in row:
            raise ValueError("Row missing required x/y fields.")

        x_val = float(row[x_field])
        y_val = float(row[y_field])
        team_value = str(row.get(team_field, "")) if team_field else ""

        if split_sides and team_field:
            if left_teams:
                if team_value in left_teams:
                    x_val = pitch_length - x_val
            elif right_teams:
                if team_value not in right_teams:
                    x_val = pitch_length - x_val
            else:
                if default_right_team is None:
                    default_right_team = team_value
                if team_value != default_right_team:
                    x_val = pitch_length - x_val

        entry = {
            "x": x_val,
            "y": y_val,
        }
        if marker_field:
            entry[marker_field] = row.get(marker_field)
        if team_field:
            entry[team_field] = team_value
        for extra_key in extras_fields:
            entry[extra_key] = row.get(extra_key)

        group_key = "Series"
        if grouping:
            if grouping not in row:
                raise ValueError(f"group_by key '{grouping}' missing in row.")
            group_key = str(row[grouping])
        elif team_field:
            group_key = team_value or "Series"
        groups[group_key].append(entry)

    primitives = {"scatter": []}
    default_palette = [
        "#ff6b6b",
        "#4ecdc4",
        "#1a535c",
        "#ffa36c",
        "#c77dff",
        "#ffd93d",
        "#6bcf99",
        "#b08ea2",
    ]

    legend_seen = set()

    for idx, (group_name, points) in enumerate(groups.items()):
        base_color = color_map_lower.get(group_name.lower(), color_map.get(group_name, default_palette[idx % len(default_palette)] if grouping else default_color))
        if marker_field:
            for pt in points:
                marker_value = str(pt.get(marker_field, ""))
                marker_key = marker_value.lower()
                marker = marker_map_lower.get(marker_key, marker_map.get(marker_value, default_marker))
                size = float(size_map_lower.get(marker_key, size_map.get(marker_value, default_size)))
                alpha = float(alpha_map_lower.get(marker_key, alpha_map.get(marker_value, default_alpha)))
                color_key_source = marker_value
                if color_field and color_field in pt:
                    color_key_source = str(pt[color_field])
                elif team_field:
                    color_key_source = pt.get(team_field, group_name)
                color_lookup = str(color_key_source).lower()
                color = color_map_lower.get(color_lookup, color_map.get(color_key_source, base_color))
                legend_params = {
                    "group": group_name,
                    "marker": marker_value or "value",
                    "team": pt.get(team_field, ""),
                    "color": color,
                    "result": marker_value or "",
                }
                if legend_field and legend_field in pt:
                    legend_params["legend"] = pt.get(legend_field)
                legend_label = legend_format.format(**legend_params)
                if legend_label in legend_seen:
                    legend_label_output = ""
                else:
                    legend_seen.add(legend_label)
                    legend_label_output = legend_label
                primitives["scatter"].append(
                    {
                        "x": [pt["x"]],
                        "y": [pt["y"]],
                        "color": color,
                        "marker": marker,
                        "s": size,
                        "alpha": alpha,
                        **({"label": legend_label_output} if legend_label_output else {}),
                    }
                )
        else:
            xs = [pt["x"] for pt in points]
            ys = [pt["y"] for pt in points]
            marker = mapping.get("marker", default_marker)
            size = float(mapping.get("size", default_size))
            alpha = float(mapping.get("alpha", default_alpha))
            color_key_source = group_name
            if color_field and points and color_field in points[0]:
                color_key_source = str(points[0][color_field])
            color_lookup = str(color_key_source).lower()
            color = color_map_lower.get(color_lookup, color_map.get(color_key_source, base_color))
            legend_params = {
                "group": group_name,
                "marker": "",
                "team": points[0].get(team_field, "") if points else "",
                "color": color,
                "result": "",
            }
            legend_label = legend_format.format(**legend_params)
            if legend_label in legend_seen:
                legend_label_output = ""
            else:
                legend_seen.add(legend_label)
                legend_label_output = legend_label
            primitives["scatter"].append(
                {
                    "x": xs,
                    "y": ys,
                    "color": color,
                    "marker": marker,
                    "s": size,
                    "alpha": alpha,
                    **({"label": legend_label_output} if legend_label_output else {}),
                }
            )

    for extra in mapping.get("extras_scatter", []):
        primitives["scatter"].append(extra)

    for key in ["lines", "arrows", "polygons", "annotations", "images"]:
        if key in mapping:
            primitives[key] = mapping[key]

    return primitives


async def _call_mpl_draw(primitives: dict, layout: dict, title: str, save_as: str) -> tuple:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "draw_pitch",
            "arguments": {
                "layout": json.dumps(layout),
                "primitives": json.dumps(primitives),
                "title": title,
                "save_as": save_as,
            },
        },
    }
    return True, payload


@mcp.tool()
async def prepare_visual_payload(dataset: str = "", mapping: str = "", target: str = "") -> str:
    """Reshape dataset into a visualization payload without rendering."""
    ok_data, table = _load_json(dataset, "dataset")
    if not ok_data:
        return table
    ok_map, map_spec = _load_json(mapping, "mapping")
    if not ok_map:
        return map_spec
    target_key = (target or "").strip().lower()

    try:
        if target_key == "mplsoccer":
            primitives = _mpl_primitives_from_dataset(table, map_spec)
            return _format_json_output({"layout": map_spec.get("layout", MPL_PITCH_DEFAULT), "primitives": primitives}, "Prepared mplsoccer primitives.")
        if target_key == "quickchart":
            ok_chart, chart = _tabular_to_chartjs(table, map_spec)
            if not ok_chart:
                return chart
            return _format_json_output(chart, "Prepared QuickChart config.")
        return "❌ Error: target must be 'mplsoccer' or 'quickchart'."
    except Exception as exc:
        logger.error("prepare_visual_payload failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def run_pipeline(plan: str = "") -> str:
    """Execute a data->visual pipeline (SQL query, transform, optional render)."""
    ok_plan, pipeline = _load_json(plan, "plan")
    if not ok_plan:
        return pipeline

    try:
        sql = pipeline.get("sql", "").strip()
        dataset = pipeline.get("dataset")
        if sql:
            success, response = await _supabase_query(sql)
            if not success:
                return response
            dataset = response
        if not dataset:
            return "❌ Error: No dataset provided in plan (and sql missing)."
        if not isinstance(dataset, dict) or "rows" not in dataset:
            return "❌ Error: Dataset must resemble Supabase SQL output (columns/rows)."

        target = pipeline.get("target", "").lower()
        mapping = pipeline.get("mapping")
        if not mapping:
            return "❌ Error: plan.mapping is required."
        primitives = chart_config = None
        if target == "mplsoccer":
            primitives = _mpl_primitives_from_dataset(dataset["rows"], mapping)
            layout = mapping.get("layout", MPL_PITCH_DEFAULT)
            if pipeline.get("render", False):
                _, payload = await _call_mpl_draw(primitives, layout, pipeline.get("title", ""), pipeline.get("save_as", ""))
                return _format_json_output(
                    {
                        "invoke": payload,
                        "note": "Send this request to the mplsoccer_viz server via tools/call to render the image."
                    },
                    "Prepared draw_pitch invocation."
                )
            return _format_json_output({"layout": layout, "primitives": primitives}, "Prepared mplsoccer primitives.")

        if target == "quickchart":
            ok_chart, chart_config = _tabular_to_chartjs(dataset, mapping)
            if not ok_chart:
                return chart_config
            if pipeline.get("render", False):
                success, image_bytes = await _quickchart_render(
                    chart_config,
                    mapping.get("format", "png"),
                    mapping.get("width", ""),
                    mapping.get("height", ""),
                    mapping.get("background", ""),
                )
                if not success:
                    return image_bytes
                fmt = mapping.get("format", "png")
                return _encode_image(image_bytes, f"image/{fmt}", "QuickChart image rendered.")
            return _format_json_output(chart_config, "Prepared QuickChart configuration.")

        return "❌ Error: plan.target must be 'mplsoccer' or 'quickchart'."
    except Exception as exc:
        logger.error("run_pipeline failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


if __name__ == "__main__":
    logger.info("Starting Analytics Coordinator MCP server...")
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        logger.error("Server error: %s", exc, exc_info=True)
        sys.exit(1)
