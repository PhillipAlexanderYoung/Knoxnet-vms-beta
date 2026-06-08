"""Pure helpers for Events FTS search query building (no OpenCV dependency)."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

_FTS_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def fts_and_query(raw: str) -> str:
    """
    Build an FTS5 MATCH expression requiring all query tokens to match (AND).
    Quoted tokens avoid FTS5 operator parsing issues.
    """
    q = (raw or "").strip()
    if not q:
        return ""
    tokens = [t for t in _FTS_TOKEN_RE.findall(q.lower()) if len(t) >= 2]
    if not tokens:
        tokens = [t for t in _FTS_TOKEN_RE.findall(q.lower()) if t]
    if not tokens:
        return q.replace('"', '""')
    if len(tokens) == 1:
        return f'"{tokens[0]}"'
    return " AND ".join(f'"{t}"' for t in tokens)


def fts_query_tokens(raw: str) -> List[str]:
    """Tokenize a user query for LIKE fallback (all tokens must match)."""
    q = (raw or "").strip()
    tokens = [t for t in _FTS_TOKEN_RE.findall(q.lower()) if t]
    return tokens if tokens else ([q.lower()] if q else [])


def merge_operator_shape_tags(
    tags: List[str],
    *,
    shape_name: Optional[str],
    operator_aliases: Optional[Sequence[str]] = None,
) -> List[str]:
    """Merge operator zone/line/tag names into searchable tags."""
    out = [str(t).strip().lower() for t in (tags or []) if str(t).strip()]
    if isinstance(shape_name, str) and shape_name.strip():
        name = shape_name.strip()
        out.append(name.lower())
        out.extend(t.lower() for t in re.split(r"[\s\-_/]+", name) if t.strip())
    if operator_aliases:
        out.extend(str(a).strip().lower() for a in operator_aliases if str(a).strip())
    return list(dict.fromkeys([t for t in out if t]))[:32]
