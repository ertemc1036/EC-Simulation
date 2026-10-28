"""OASIS reply / interaction API facts for Phase 4/5.

camel-oasis==0.2.5 (Reddit platform) has **no** nested-reply field and **no**
separate ``reply`` ActionType. Documented here so the engine, prompts, graph,
and dashboard stay aligned with the real platform surface.

Phase 5 adds *local* comment references in content (``→Kommentar#N``) because
OASIS cannot nest comments; graph ``responds_to`` uses those refs as evidence.
"""
from __future__ import annotations

from typing import Any

from backend.simulation.social_dynamics import SOCIAL_RELATIONS

# Verified against oasis.social_agent.agent_action.SocialAction.create_comment
# and oasis.social_platform.platform.Platform.create_comment + schema/comment.sql.
OASIS_REPLY_API: dict[str, Any] = {
    "package": "camel-oasis",
    "parent_comment_id_supported": False,
    "separate_reply_action_type": False,
    "create_comment_signature": "create_comment(post_id: int, content: str)",
    "comment_schema_columns": [
        "comment_id",
        "post_id",
        "user_id",
        "content",
        "created_at",
        "num_likes",
        "num_dislikes",
    ],
    "documented_alternative": (
        "OASIS has no parent_comment_id. Reply = CREATE_COMMENT on an observed "
        "post_id (flat thread under the post). Nested comment→comment replies "
        "are not supported — use local text refs →Kommentar#N in content. "
        "Agreement / disagreement = LIKE_POST / DISLIKE_POST "
        "or LIKE_COMMENT / DISLIKE_COMMENT on the observed id — no quotas; "
        "additionally, clear textual agree/disagree markers in a comment that "
        "already contains a resolvable →Kommentar# ref (Phase 5.2). "
        "Round 2+: prefer like_comment/dislike_comment on visible Kommentar# IDs."
    ),
    "local_comment_ref_convention": "→Kommentar#<id> inside create_comment content",
    "interaction_tools": [
        "create_comment",  # reply to post
        "like_post",
        "dislike_post",
        "like_comment",
        "dislike_comment",
        "create_post",
        "follow",
        "do_nothing",
    ],
    "graph_relations_evidenced_only": sorted(
        {"replied_to", *SOCIAL_RELATIONS}
    ),
}


def oasis_reply_status() -> dict[str, Any]:
    """Public payload for health / result meta (no secrets)."""
    return {
        "parent_comment_id_supported": OASIS_REPLY_API["parent_comment_id_supported"],
        "separate_reply_action_type": OASIS_REPLY_API["separate_reply_action_type"],
        "reply_via": "create_comment(post_id, content)",
        "documented_alternative": OASIS_REPLY_API["documented_alternative"],
        "graph_relations_evidenced_only": list(
            OASIS_REPLY_API["graph_relations_evidenced_only"]
        ),
    }
