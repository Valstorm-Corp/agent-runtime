"""Unit tests for Valstorm Platform Tools registered in ToolRegistry."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock

from core.tools import ToolRegistry
from tools.valstorm_client import ValstormApiClient
from tools.valstorm_tools import register_valstorm_tools


@pytest.fixture
def mock_api_client():
    mock_sql_resp = httpx.Response(
        status_code=200,
        json={"records": [{"id": "cont_99", "name": "Jane Smith", "status": "Active"}]},
        request=httpx.Request("POST", "http://localhost:8000/query"),
    )
    mock_vfs_resp = httpx.Response(
        status_code=200,
        json={"results": [{"file_id": "file_88", "filename": "Contract.pdf", "rrf_score": 0.89, "sources": ["qdrant"]}]},
        request=httpx.Request("POST", "http://localhost:8000/v1/search"),
    )
    mock_schema_resp = httpx.Response(
        status_code=200,
        json={"contact": {"title": "Contact", "prefix": "cont"}},
        request=httpx.Request("GET", "http://localhost:8000/schema"),
    )

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(
        side_effect=[mock_sql_resp, mock_vfs_resp, mock_schema_resp]
    )

    http_client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    return ValstormApiClient(token="mock_token", client=http_client)


@pytest.mark.asyncio
async def test_valstorm_tools_registration_and_execution(mock_api_client):
    registry = ToolRegistry()
    register_valstorm_tools(registry=registry, client=mock_api_client)

    tool_names = registry.list_tools()
    assert "valstorm_sql_query" in tool_names
    assert "valstorm_mongo_query" in tool_names
    assert "valstorm_vfs_search" in tool_names
    assert "valstorm_vfs_browse" in tool_names
    assert "valstorm_vfs_get_file" in tool_names
    assert "valstorm_vfs_write_file" in tool_names
    assert "valstorm_record_cud" in tool_names
    assert "valstorm_record_merge" in tool_names
    assert "valstorm_records_hydrate_batch" in tool_names
    assert "valstorm_schema_inspect" in tool_names
    assert "valstorm_function_list" in tool_names
    assert "valstorm_function_call" in tool_names
    assert "valstorm_validate_function" in tool_names
    assert "valstorm_get_execution_logs" in tool_names
    assert "valstorm_get_log_detail" in tool_names

    # Check schemas
    schemas = registry.get_schemas()
    assert len(schemas) == len(tool_names)
    schema_map = {s["name"]: s for s in schemas}
    assert "query" in schema_map["valstorm_sql_query"]["parameters"]["properties"]
    assert "api_name" in schema_map["valstorm_record_cud"]["parameters"]["properties"]
    assert "master_record_id" in schema_map["valstorm_record_merge"]["parameters"]["properties"]
    assert "channel" in schema_map["slack_post_message"]["parameters"]["properties"]

    # Execute SQL Tool
    sql_res = await registry.execute_async(
        "valstorm_sql_query",
        {"query": "SELECT id, name FROM contact WHERE status = 'Active'"}
    )
    assert not sql_res.is_error
    assert "Jane Smith" in sql_res.output
    assert sql_res.duration_ms >= 0.0

    # Execute VFS Search Tool
    vfs_res = await registry.execute_async(
        "valstorm_vfs_search",
        {"query": "Contract", "limit": 5}
    )
    assert not vfs_res.is_error
    assert "Contract.pdf" in vfs_res.output

    # Execute Schema Inspect Tool
    schema_res = await registry.execute_async(
        "valstorm_schema_inspect",
        {"api_name": ""}
    )
    assert not schema_res.is_error
    assert "contact" in schema_res.output
