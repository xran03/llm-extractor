"""Robust JSON parsing of model output.

Even with strict structured output, gateways differ: some wrap the payload in
code fences, some return a bare array, some return ``{"records": [...]}``, and
a truncated completion yields invalid JSON. This module recovers records from
all of those, including salvaging complete objects out of a cut-off array, and
closing a cut-off object — a partial result beats losing a whole document.
"""
from __future__ import annotations

import json
import re


def strip_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_json_object(text: str) -> dict:
    """Return the first JSON object found in ``text`` (``{}`` when none).

    A completion cut off at the output-token limit is salvaged rather than
    discarded: a figure holding 300 points is exactly the one worth keeping
    partially, and returning ``{}`` would report it as an empty figure.
    """
    s = strip_fences(text)
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start: end + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    if start != -1:
        obj = close_truncated_json(s[start:])
        if isinstance(obj, dict):
            obj[TRUNCATED_KEY] = True
            return obj
    return {}


#: Marks a payload recovered from a cut-off completion, so callers can report
#: it instead of mistaking a partial reading for a complete one.
TRUNCATED_KEY = "_truncated"


def close_truncated_json(s: str):
    """Parse JSON that was cut off mid-emission by closing what is still open.

    The text is rewound to the last position at which every value already
    emitted was complete, and the open containers are then closed in order.
    Returns ``None`` when nothing can be recovered.
    """
    stack: list = []
    safe: tuple | None = None      # (cut index, containers open at that point)
    in_string = False
    escaped = False

    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = (i + 1, tuple(stack))
        elif ch == ",":
            # Everything before the separator is a complete element.
            safe = (i, tuple(stack))

    if safe is None:
        return None
    cut, open_containers = safe
    candidate = s[:cut] + "".join(reversed(open_containers))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def extract_json_array(text: str, key: str = "records") -> list:
    """Return a list of records from a model response.

    Accepts a bare array, a fenced block, an object wrapping a list (preferring
    ``key``), or prose containing a ``[ ... ]`` block.
    """
    if text is None:
        raise ValueError("empty model output")
    s = strip_fences(text)
    if not s:
        return []

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get(key), list):
            return obj[key]
        for value in obj.values():
            if isinstance(value, list):
                return value
        return [obj]

    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(s[start: end + 1])
        except json.JSONDecodeError:
            pass

    # Salvage a truncated array (model hit the output-token limit): recover
    # every complete top-level object that was emitted before the cut.
    if start != -1:
        salvaged = salvage_objects(s[start:])
        if salvaged:
            return salvaged
    raise ValueError("no JSON array found in model output")


def salvage_objects(s: str) -> list:
    """Recover complete top-level ``{...}`` objects from truncated JSON."""
    objects: list = []
    depth = 0
    in_string = False
    escaped = False
    start = None
    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objects.append(json.loads(s[start: i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return objects
