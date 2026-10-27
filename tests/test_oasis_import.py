"""Verify OASIS + mcp pin without LLM calls."""
from __future__ import annotations

import importlib.metadata


def test_mcp_is_1x():
    version = importlib.metadata.version("mcp")
    major = int(version.split(".")[0])
    assert major == 1, f"mcp must be 1.x for camel-oasis; got {version}"


def test_import_oasis_and_core_symbols():
    import oasis
    from oasis import (
        ActionType,
        AgentGraph,
        DefaultPlatformType,
        LLMAction,
        ManualAction,
        SocialAgent,
        UserInfo,
        make,
    )

    assert oasis.__version__ == "0.2.5"
    assert callable(make)
    assert ManualAction is not None
    assert LLMAction is not None
    assert AgentGraph is not None
    assert SocialAgent is not None
    assert UserInfo is not None
    assert DefaultPlatformType.REDDIT
    assert ActionType.CREATE_POST
