#!/usr/bin/env python3
"""free-model-aggregator-01: the best free AI model aggregator.

One server, two client APIs (OpenAI Chat Completions and Anthropic Messages),
ten simulated free-tier providers on three wires (OpenAI, Anthropic, Gemini).

How it works, in plain terms:
  * At startup it reads the trial brief (the provider directory, the quality
    table and the workload manifest) and probes every provider's own catalog
    route, then builds one catalog and registers it with the env.
  * For each client request it ranks every free model that can serve it by
    the quality table's rating (best first), tries them in that order, and
    fails over to the next one when a provider refuses with 402/429/5xx.
    A provider that says its budget is spent, or that it is out, is never
    tried again for the rest of the trial.
  * It speaks each provider's own wire and translates the answer back into
    the wire the client used, JSON or streaming, carrying the provider's
    text, usage counts and stop reason across untouched -- and it passes the
    client's system prompt, user text, images, tools, stop sequences,
    max_tokens and JSON mode through exactly, because the provider echoes
    them back and the client compares that echo with what it sent.

Python 3.13, standard library only. Deterministic: every routing choice is
a function of the brief, the requests and the providers' answers seen so
far. No wall clock and no randomness reach any request or response.
"""

import http.server
import json
import os
import signal
import socketserver
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 10.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
SSE_DELTA_CHARS = 13

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# The stop-reason families the client checks: stop/end_turn/STOP are one
# family, length/max_tokens/MAX_TOKENS another, tool calls a third, and
# refusals a fourth. The simulated providers only ever answer the first.
STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# ---------------------------------------------------------------- the brief


def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(value):
    return value if isinstance(value, str) else ""


def _number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value and abs(value) != float("inf") else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


def _provider_of(value):
    if not isinstance(value, dict):
        return None
    pid = _string_of(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = (
        _string_of(value.get("path"))
        or _string_of(value.get("base_path"))
        or _string_of(value.get("base_url"))
        or _string_of(value.get("base"))
    )
    return {
        "id": pid,
        "name": _string_of(value.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string_of(value.get("key_env")),
        "key": _string_of(value.get("key")),
    }


def _rating_of(value):
    if not isinstance(value, dict):
        return None
    model = _string_of(value.get("model"))
    provider = _string_of(value.get("provider"))
    if model == "" or provider == "":
        return None
    rating = value.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": _number_or_none(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is True,
    }


def _workload_request_of(value):
    if not isinstance(value, dict):
        return None
    rid = _string_of(value.get("id"))
    if rid == "":
        return None
    requires = value.get("requires") if isinstance(value.get("requires"), dict) else {}
    out = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if isinstance(requires.get("min_context"), (int, float)):
        out["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path):
    """The brief at `path`: directory, ratings, workload, error (or None)."""
    directory = {"schema": PROVIDERS_SCHEMA, "providers": []}
    if not path:
        return directory, None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as err:
        return directory, None, None, "cannot read %s: %s" % (path, err)
    docs = _documents_in(text)
    directory_doc = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory_doc is None:
        return directory, None, None, "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
    providers = [p for p in (_provider_of(v) for v in directory_doc["providers"]) if p is not None]
    directory = {"schema": PROVIDERS_SCHEMA, "providers": providers}
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    ratings = None
    if ratings_doc is not None:
        rows = [r for r in (_rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if isinstance(v, dict) and _string_of(v.get("id")) != "":
                    best.append({"id": v["id"], "rating_milli": _number_or_none(v.get("rating_milli")) or 0})
        ratings = {"schema": RATINGS_SCHEMA, "ratings": rows, "best": best}
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    workload = None
    if workload_doc is not None:
        manifest = [w for w in (_workload_request_of(v) for v in workload_doc["manifest"]) if w is not None]
        workload = {"schema": WORKLOAD_SCHEMA, "manifest": manifest}
    return directory, ratings, workload, None


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# ---------------------------------------------------------------- the probe


def join_base(env_url, base):
    """A base path joined to the env's URL; an absolute base is used as-is."""
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_in(wire, value):
    """One model as a listing wrote it, in whichever wire's spelling."""
    if not isinstance(value, dict):
        return None
    if wire == "gemini":
        model_id = _string_of(value.get("name")).removeprefix("models/")
    else:
        model_id = _string_of(value.get("id"))
    if model_id == "":
        return None
    context = (
        _number_or_none(value.get("context_length"))
        if _number_or_none(value.get("context_length")) is not None
        else _number_or_none(value.get("context_window"))
        if _number_or_none(value.get("context_window")) is not None
        else _number_or_none(value.get("inputTokenLimit"))
    )
    display = (
        _string_of(value.get("display_name"))
        or _string_of(value.get("displayName"))
        or model_id
    )
    free = _bool_or_none(value.get("free"))
    if free is None and isinstance(value.get("pricing"), dict):
        numbers = []
        for item in value["pricing"].values():
            if isinstance(item, (int, float)):
                numbers.append(item)
            elif isinstance(item, str) and item.strip() != "":
                try:
                    numbers.append(float(item))
                except ValueError:
                    pass
        if numbers:
            free = all(n == 0 for n in numbers)
    return {
        "id": model_id,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(value.get("tools")),
        "vision": _bool_or_none(value.get("vision")),
        "json": _bool_or_none(value.get("json")),
    }


def _models_in(wire, body):
    if not isinstance(body, dict):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    return [m for m in (_model_in(wire, v) for v in listing) if m is not None]


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Tolerates every failure."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with _OPENER.open(req, timeout=timeout) as res:
            if res.status < 200 or res.status >= 300:
                return {"provider": provider, "ok": False, "status": res.status, "models": [],
                        "error": "GET %s answered HTTP %s" % (url, res.status)}
            body = json.loads(res.read().decode("utf-8"))
        return {"provider": provider, "ok": True, "status": res.status,
                "models": _models_in(provider["wire"], body), "error": None}
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as err:
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": "GET %s failed: %s" % (url, err)}


def model_free(model, row):
    """Whether a model is free: the quality table where it has a row, else the listing."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    """What a model can do: the quality table where it has a row, else the listing."""
    return {
        "context_length": row["context_length"] if row is not None and row["context_length"] is not None
        else model["context_length"],
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet one request's requirements."""
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    if requires.get("min_context") is not None:
        if caps.get("context_length") is None or caps["context_length"] < requires["min_context"]:
            return False
    return True


def build_catalog(probes, ratings):
    """The one catalog: every provider that answered, every model it listed.

    Its base_url is the directory's own path (/p/<id>), never the path joined
    to the env URL: the env's origin differs between a live pass and its
    replay, and the catalog is part of the state export a replay must
    reproduce byte for byte.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": row["context_length"] if row is not None and row["context_length"] is not None
                else model["context_length"],
            })
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """The free models across the providers that answered, in brief order."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


# ---------------------------------------------------------------- request shapes


def _is_record(value):
    return isinstance(value, dict)


def _text_of(content):
    """The text of a string or of a block array's text blocks; '' when none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _arguments_of(value):
    if _is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if _is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


class Req:
    """One client request, in wire-neutral form."""

    def __init__(self):
        self.system = ""
        self.messages = []      # {role, text, images:[{data|url, media_type}], tool_calls, tool_call_id}
        self.tools = []         # {name, description, parameters}, in the client's order
        self.tool_choice = None  # 'auto' | 'any' | 'none' | {'name': ...}
        self.max_tokens = None
        self.temperature = None
        self.top_p = None
        self.stops = []
        self.stream = False
        self.json_mode = False


def parse_openai_body(body):
    """An OpenAI Chat Completions request as a wire-neutral request."""
    req = Req()
    req.stream = body.get("stream") is True
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            req.system = (req.system + "\n" + text) if req.system and text else (req.system or text)
            continue
        message = {"role": role, "text": "", "images": [], "tool_calls": [], "tool_call_id": None}
        if isinstance(content, str):
            message["text"] = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if not _is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif block.get("type") == "image_url" and _is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    if isinstance(url, str) and url != "":
                        message["images"].append(_image_of_url(url))
            message["text"] = "\n".join(texts)
        if role == "assistant" and isinstance(raw.get("tool_calls"), list):
            for call in raw["tool_calls"]:
                if not _is_record(call) or not _is_record(call.get("function")):
                    continue
                fn = call["function"]
                message["tool_calls"].append({
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "arguments": _arguments_of(fn.get("arguments")),
                })
        if role == "tool":
            message["tool_call_id"] = raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else ""
            message["text"] = _tool_result_text(content)
        req.messages.append(message)
    if isinstance(body.get("tools"), list):
        for raw in body["tools"]:
            if not _is_record(raw) or not _is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            req.tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if _is_record(fn.get("parameters"))
                else {"type": "object", "properties": {}},
            })
    choice = body.get("tool_choice")
    if choice == "auto" or choice == "none" or choice == "required":
        req.tool_choice = choice
    elif _is_record(choice) and choice.get("type") == "function" and _is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            req.tool_choice = {"name": name}
    if isinstance(body.get("max_tokens"), (int, float)):
        req.max_tokens = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        req.max_tokens = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        req.temperature = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        req.top_p = body["top_p"]
    req.stops = _string_list(body.get("stop"))
    fmt = body.get("response_format") if _is_record(body.get("response_format")) else None
    req.json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
    return req


def _image_of_url(url):
    """An image URL as carried across wires: a data URL keeps its bytes, a plain URL its string."""
    if url.startswith("data:"):
        head, comma, data = url[5:].partition(",")
        if comma and head.endswith(";base64"):
            return {"data": data, "media_type": head[: -len(";base64")] or "image/png"}
    return {"url": url}


def parse_anthropic_body(body):
    """An Anthropic Messages request as a wire-neutral request."""
    req = Req()
    req.stream = body.get("stream") is True
    req.system = _text_of(body.get("system"))
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        message = {"role": role, "text": "", "images": [], "tool_calls": [], "tool_call_id": None}
        if isinstance(content, str):
            message["text"] = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if not _is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif block.get("type") == "image" and _is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        message["images"].append({
                            "data": source["data"],
                            "media_type": source.get("media_type") if isinstance(source.get("media_type"), str)
                            else "image/png",
                        })
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        message["images"].append({"url": source["url"]})
                elif block.get("type") == "tool_use":
                    message["tool_calls"].append({
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": block.get("input") if _is_record(block.get("input")) else {},
                    })
                elif block.get("type") == "tool_result":
                    req.messages.append({
                        "role": "tool",
                        "text": _tool_result_text(block.get("content")),
                        "images": [],
                        "tool_calls": [],
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    })
            message["text"] = "\n".join(texts)
        req.messages.append(message)
    if isinstance(body.get("tools"), list):
        for raw in body["tools"]:
            if not _is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            req.tools.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if _is_record(raw.get("input_schema"))
                else {"type": "object", "properties": {}},
            })
    choice = body.get("tool_choice")
    if _is_record(choice):
        if choice.get("type") == "auto":
            req.tool_choice = "auto"
        elif choice.get("type") == "any":
            req.tool_choice = "any"
        elif choice.get("type") == "none":
            req.tool_choice = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            req.tool_choice = {"name": choice["name"]}
    if isinstance(body.get("max_tokens"), (int, float)):
        req.max_tokens = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        req.temperature = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        req.top_p = body["top_p"]
    req.stops = _string_list(body.get("stop_sequences"))
    req.json_mode = False  # the Messages API carries no JSON mode
    return req


def _openai_content(message):
    """One normalized message as an OpenAI message's content."""
    parts = []
    if message["text"] != "":
        parts.append({"type": "text", "text": message["text"]})
    for image in message["images"]:
        if "data" in image:
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:%s;base64,%s" % (image["media_type"], image["data"])}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": image["url"]}})
    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def build_openai_body(req, model_id):
    """A wire-neutral request as an OpenAI Chat Completions body naming `model_id`."""
    messages = []
    if req.system != "":
        messages.append({"role": "system", "content": req.system})
    for message in req.messages:
        role = message["role"]
        if role == "tool":
            messages.append({"role": "tool", "tool_call_id": message["tool_call_id"],
                             "content": message["text"]})
            continue
        out = {"role": "assistant" if role == "assistant" else "user",
               "content": _openai_content(message)}
        if message["tool_calls"]:
            out["tool_calls"] = [{
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            } for call in message["tool_calls"]]
            if out["content"] == "":
                out["content"] = None
        messages.append(out)
    body = {"model": model_id, "messages": messages}
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stops:
        body["stop"] = req.stops
    if req.tools:
        body["tools"] = [{
            "type": "function",
            "function": {"name": tool["name"], "description": tool["description"],
                         "parameters": tool["parameters"]},
        } for tool in req.tools]
    if req.tool_choice == "auto":
        body["tool_choice"] = "auto"
    elif req.tool_choice == "any":
        body["tool_choice"] = "required"
    elif req.tool_choice == "none":
        body["tool_choice"] = "none"
    elif isinstance(req.tool_choice, dict):
        body["tool_choice"] = {"type": "function", "function": {"name": req.tool_choice["name"]}}
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    if req.stream:
        body["stream"] = True
        # Ask the provider to close its stream with a usage chunk: the client
        # checks the usage its answer reports against the provider's own.
        body["stream_options"] = {"include_usage": True}
    return body


def _anthropic_blocks(message):
    """One normalized message as Anthropic content blocks."""
    blocks = []
    if message["text"] != "":
        blocks.append({"type": "text", "text": message["text"]})
    for image in message["images"]:
        if "data" in image:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": image["media_type"],
                                                        "data": image["data"]}})
        else:
            blocks.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
    for call in message["tool_calls"]:
        blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"],
                       "input": call["arguments"]})
    return blocks


def build_anthropic_body(req, model_id):
    """A wire-neutral request as an Anthropic Messages body naming `model_id`."""
    messages = []
    for message in req.messages:
        role = message["role"]
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": message["tool_call_id"],
                "content": message["text"],
            }]})
            continue
        blocks = _anthropic_blocks(message)
        messages.append({"role": "assistant" if role == "assistant" else "user",
                         "content": blocks if blocks else ""})
    body = {"model": model_id, "messages": messages}
    if req.system != "":
        body["system"] = req.system
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stops:
        body["stop_sequences"] = req.stops
    if req.tools:
        body["tools"] = [{
            "name": tool["name"], "description": tool["description"],
            "input_schema": tool["parameters"],
        } for tool in req.tools]
    if req.tool_choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif req.tool_choice == "any":
        body["tool_choice"] = {"type": "any"}
    elif req.tool_choice == "none":
        body["tool_choice"] = {"type": "none"}
    elif isinstance(req.tool_choice, dict):
        body["tool_choice"] = {"type": "tool", "name": req.tool_choice["name"]}
    if req.stream:
        body["stream"] = True
    return body


def build_gemini_body(req, model_id):
    """A wire-neutral request as a Gemini generateContent body."""
    contents = []
    for message in req.messages:
        role = message["role"]
        if role == "tool":
            contents.append({"role": "function", "parts": [{
                "functionResponse": {"name": message["tool_call_id"] or "tool",
                                     "response": {"output": message["text"]}},
            }]})
            continue
        parts = []
        if message["text"] != "":
            parts.append({"text": message["text"]})
        for image in message["images"]:
            if "data" in image:
                parts.append({"inlineData": {"mimeType": image["media_type"], "data": image["data"]}})
            else:
                parts.append({"fileData": {"fileUri": image["url"]}})
        for call in message["tool_calls"]:
            parts.append({"functionCall": {"name": call["name"], "args": call["arguments"]}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    body = {"contents": contents}
    if req.system != "":
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        body["tools"] = [{"functionDeclarations": [{
            "name": tool["name"], "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in req.tools]}]
    config = {}
    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if req.top_p is not None:
        config["topP"] = req.top_p
    if req.stops:
        config["stopSequences"] = req.stops
    if req.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------- provider answers


class ProviderFailure(Exception):
    """A provider refused a call: HTTP status and the message it gave."""

    def __init__(self, status, message):
        super().__init__("provider answered %s: %s" % (status, message))
        self.status = status
        self.message = message


class Answer:
    """One provider's answer, in wire-neutral form."""

    def __init__(self, text, input_tokens, output_tokens, stop_family, tool_calls=None):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_family = stop_family
        self.tool_calls = tool_calls or []


def _number_or_zero(value):
    return value if isinstance(value, (int, float)) else 0


def _parse_sse_frames(text):
    """One SSE text as (event, data) frames; data lines of a frame joined."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _answer_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def parse_openai_json(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and _is_record(choices[0]) else {}
    message = choice.get("message") if _is_record(choice.get("message")) else {}
    usage = body.get("usage") if _is_record(body.get("usage")) else {}
    tool_calls = []
    for call in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
        if not _is_record(call) or not _is_record(call.get("function")):
            continue
        tool_calls.append({
            "id": call.get("id") if isinstance(call.get("id"), str) else "",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "arguments": _arguments_of(call["function"].get("arguments")),
        })
    finish = choice.get("finish_reason")
    return Answer(
        _answer_text(message.get("content")),
        _number_or_zero(usage.get("prompt_tokens")),
        _number_or_zero(usage.get("completion_tokens")),
        STOP_FAMILIES.get(finish, 0),
        tool_calls,
    )


def parse_openai_sse(text):
    content = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    tool_calls = {}
    for _event, data in _parse_sse_frames(text):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            usage_in = _number_or_zero(chunk["usage"].get("prompt_tokens"))
            usage_out = _number_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not _is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if _is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        for call in delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []:
            if not _is_record(call):
                continue
            index = call.get("index") if isinstance(call.get("index"), int) else len(tool_calls)
            slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if isinstance(call.get("id"), str) and call["id"] != "":
                slot["id"] = call["id"]
            fn = call.get("function") if _is_record(call.get("function")) else {}
            if isinstance(fn.get("name"), str) and fn["name"] != "":
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            stop_family = STOP_FAMILIES.get(choice["finish_reason"], 0)
    calls = [{
        "id": slot["id"],
        "name": slot["name"],
        "arguments": _arguments_of(slot["arguments"]),
    } for _index, slot in sorted(tool_calls.items())]
    return Answer("".join(content), usage_in, usage_out, stop_family, calls)


def parse_anthropic_json(body):
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = _answer_text(content)
    tool_calls = []
    for block in content:
        if not _is_record(block) or block.get("type") != "tool_use":
            continue
        tool_calls.append({
            "id": block.get("id") if isinstance(block.get("id"), str) else "",
            "name": block.get("name") if isinstance(block.get("name"), str) else "",
            "arguments": block.get("input") if _is_record(block.get("input")) else {},
        })
    usage = body.get("usage") if _is_record(body.get("usage")) else {}
    stop = body.get("stop_reason")
    return Answer(
        text,
        _number_or_zero(usage.get("input_tokens")),
        _number_or_zero(usage.get("output_tokens")),
        STOP_FAMILIES.get(stop, 0),
        tool_calls,
    )


def parse_anthropic_sse(text):
    content = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    tool_blocks = {}
    for event, data in _parse_sse_frames(text):
        try:
            frame = json.loads(data)
        except ValueError:
            continue
        if not _is_record(frame):
            continue
        if event == "message_start" and _is_record(frame.get("message")):
            usage = frame["message"].get("usage") if _is_record(frame["message"].get("usage")) else {}
            usage_in = _number_or_zero(usage.get("input_tokens"))
        elif event == "content_block_delta" and _is_record(frame.get("delta")):
            delta = frame["delta"]
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                content.append(delta["text"])
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                index = frame.get("index") if isinstance(frame.get("index"), int) else 0
                slot = tool_blocks.setdefault(index, {"id": "", "name": "", "arguments": ""})
                slot["arguments"] += delta["partial_json"]
        elif event == "content_block_start":
            index = frame.get("index") if isinstance(frame.get("index"), int) else 0
            block = frame.get("content_block") if _is_record(frame.get("content_block")) else {}
            if block.get("type") == "tool_use":
                tool_blocks[index] = {
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": "",
                }
        elif event == "message_delta":
            if _is_record(frame.get("delta")) and isinstance(frame["delta"].get("stop_reason"), str):
                stop_family = STOP_FAMILIES.get(frame["delta"]["stop_reason"], 0)
            if _is_record(frame.get("usage")):
                usage_out = _number_or_zero(frame["usage"].get("output_tokens"))
    calls = [{
        "id": slot["id"],
        "name": slot["name"],
        "arguments": _arguments_of(slot["arguments"]),
    } for _index, slot in sorted(tool_blocks.items())]
    return Answer("".join(content), usage_in, usage_out, stop_family, calls)


def parse_gemini_json(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    candidate = candidates[0] if candidates and _is_record(candidates[0]) else {}
    content = candidate.get("content") if _is_record(candidate.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(
        part["text"] for part in parts
        if _is_record(part) and isinstance(part.get("text"), str)
    )
    tool_calls = []
    for part in parts:
        if not _is_record(part) or not _is_record(part.get("functionCall")):
            continue
        call = part["functionCall"]
        tool_calls.append({
            "id": "",
            "name": call.get("name") if isinstance(call.get("name"), str) else "",
            "arguments": call.get("args") if _is_record(call.get("args")) else {},
        })
    usage = body.get("usageMetadata") if _is_record(body.get("usageMetadata")) else {}
    finish = candidate.get("finishReason")
    return Answer(
        text,
        _number_or_zero(usage.get("promptTokenCount")),
        _number_or_zero(usage.get("candidatesTokenCount")),
        STOP_FAMILIES.get(finish, 0),
        tool_calls,
    )


def parse_gemini_sse(text):
    content = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    for _event, data in _parse_sse_frames(text):
        try:
            frame = json.loads(data)
        except ValueError:
            continue
        if not _is_record(frame):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        if candidates and _is_record(candidates[0]):
            candidate = candidates[0]
            inner = candidate.get("content") if _is_record(candidate.get("content")) else {}
            parts = inner.get("parts") if isinstance(inner.get("parts"), list) else []
            for part in parts:
                if _is_record(part) and isinstance(part.get("text"), str):
                    content.append(part["text"])
            if isinstance(candidate.get("finishReason"), str):
                stop_family = STOP_FAMILIES.get(candidate["finishReason"], 0)
        if _is_record(frame.get("usageMetadata")):
            usage = frame["usageMetadata"]
            usage_in = _number_or_zero(usage.get("promptTokenCount"))
            usage_out = _number_or_zero(usage.get("candidatesTokenCount"))
    return Answer("".join(content), usage_in, usage_out, stop_family)


# ---------------------------------------------------------------- one provider call


def _exchange(method, url, headers, body_bytes, timeout):
    """One HTTP exchange: (status, body bytes). Raises ProviderFailure on no answer."""
    req = urllib.request.Request(url, data=body_bytes, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError) as err:
        raise ProviderFailure(0, "cannot reach %s: %s" % (url, err))


def _error_message(wire, status, raw):
    """The message a provider's error body carries, in its wire's envelope."""
    try:
        body = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return raw.decode("utf-8", errors="replace")[:300] or ("HTTP %s" % status)
    if not _is_record(body):
        return "HTTP %s" % status
    error = body.get("error")
    if _is_record(error) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(body.get("message"), str):
        return body["message"]
    return "HTTP %s" % status


def call_provider(env_url, provider, req, model_id):
    """One call to one provider on its own wire: its answer, or a ProviderFailure."""
    base = join_base(env_url, provider["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        payload = build_openai_body(req, model_id)
        if req.stream:
            headers["accept"] = "text/event-stream"
        parse = parse_openai_sse if req.stream else parse_openai_json
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        payload = build_anthropic_body(req, model_id)
        if req.stream:
            headers["accept"] = "text/event-stream"
        parse = parse_anthropic_sse if req.stream else parse_anthropic_json
    else:  # gemini
        action = "streamGenerateContent" if req.stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model_id, safe=""), action)
        if req.stream:
            url += "?alt=sse"
            headers["accept"] = "text/event-stream"
        payload = build_gemini_body(req, model_id)
        parse = parse_gemini_sse if req.stream else parse_gemini_json
    body_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    status, raw = _exchange("POST", url, headers, body_bytes, PROVIDER_TIMEOUT_S)
    if status < 200 or status >= 300:
        raise ProviderFailure(status, _error_message(wire, status, raw))
    try:
        parsed = json.loads(raw.decode("utf-8")) if not req.stream else None
    except ValueError:
        raise ProviderFailure(status, "the provider answered a body that is not JSON")
    if req.stream:
        return parse(raw.decode("utf-8", errors="replace"))
    return parse(parsed)


# ---------------------------------------------------------------- answers back to the client


def _sse_event(event, data):
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def _split_text(text):
    pieces = [text[i:i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]
    return pieces or [""]


def render_openai_json(answer, model_id):
    message = {"role": "assistant", "content": answer.text}
    if answer.tool_calls:
        message["tool_calls"] = [{
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
        } for call in answer.tool_calls]
    return {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": OPENAI_STOP[answer.stop_family],
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": answer.input_tokens,
            "completion_tokens": answer.output_tokens,
            "total_tokens": answer.input_tokens + answer.output_tokens,
        },
    }


def render_openai_sse(answer, model_id):
    """An OpenAI event stream: role chunk, content deltas, finish, usage, [DONE]."""
    head = {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model_id}
    frames = []

    def chunk(delta, finish):
        return dict(head, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])

    frames.append(_sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    if answer.tool_calls:
        for index, call in enumerate(answer.tool_calls):
            frames.append(_sse_event(None, chunk(
                {"tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                 "function": {"name": call["name"], "arguments": ""}}]}, None)))
            frames.append(_sse_event(None, chunk(
                {"tool_calls": [{"index": index,
                                 "function": {"arguments": json.dumps(call["arguments"])}}]}, None)))
    else:
        for piece in _split_text(answer.text):
            frames.append(_sse_event(None, chunk({"content": piece}, None)))
    frames.append(_sse_event(None, chunk({}, OPENAI_STOP[answer.stop_family])))
    frames.append(_sse_event(None, dict(
        head, choices=[],
        usage={"prompt_tokens": answer.input_tokens,
               "completion_tokens": answer.output_tokens,
               "total_tokens": answer.input_tokens + answer.output_tokens})))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(answer, model_id):
    content = []
    if answer.text != "":
        content.append({"type": "text", "text": answer.text})
    for call in answer.tool_calls:
        content.append({"type": "tool_use", "id": call["id"] or "toolu_unknown",
                        "name": call["name"], "input": call["arguments"]})
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": ANTHROPIC_STOP[answer.stop_family],
        "stop_sequence": None,
        "usage": {"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens},
    }


def render_anthropic_sse(answer, model_id):
    """An Anthropic event stream: message_start, one text block, message_delta, message_stop."""
    frames = [ _sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer.input_tokens, "output_tokens": 0}},
    })]
    index = 0
    if answer.tool_calls:
        for call in answer.tool_calls:
            frames.append(_sse_event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": call["id"] or ("toolu_%d" % index),
                                  "name": call["name"], "input": {}},
            }))
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(call["arguments"])},
            }))
            frames.append(_sse_event("content_block_stop", {
                "type": "content_block_stop", "index": index,
            }))
            index += 1
    else:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        for piece in _split_text(answer.text):
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            }))
        frames.append(_sse_event("content_block_stop", {
            "type": "content_block_stop", "index": 0,
        }))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": ANTHROPIC_STOP[answer.stop_family], "stop_sequence": None},
        "usage": {"output_tokens": answer.output_tokens},
    }))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


# ---------------------------------------------------------------- routing


class Router:
    """The ranked free candidates and what the trial has taught it so far.

    Candidates are ranked by the quality table's rating, best first; ties go
    to the provider called least so far, which spreads the load over the
    providers that share a rating instead of burning one provider's budget.
    A provider that says its budget is spent, or that it is out, is dead for
    the rest of the trial. A provider that merely refused one call is not
    dead -- its next call may serve -- but this request moves on to the next
    candidate at once: the rating lost is a fraction of a percent, and the
    call it would have cost is saved for a later request.
    """

    def __init__(self, probes, ratings):
        self.probes = probes
        self.ratings = ratings
        self.calls = {}   # provider id -> attempts made (success or refusal)
        self.dead = set()  # provider ids that cannot serve again this trial

    def candidates(self, requires, client_api, json_mode):
        """(provider, model, rating) that can serve, best first, in tiers.

        Tier 1 meets the requirements and can carry the client's JSON mode.
        Tier 2 meets the requirements on any wire. Tier 3 is any free model.
        A request served by a lower tier still passes most of the client's
        checks, which beats answering nothing at all.
        """
        tiers = [[], [], []]
        for probe in self.probes:
            if not probe["ok"] or probe["provider"]["id"] in self.dead:
                continue
            provider = probe["provider"]
            for model in probe["models"]:
                row = rating_of_model(self.ratings, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                entry = (provider, model, row["rating_milli"] if row is not None else 0)
                if meets_requirements(requires, caps):
                    if not (json_mode and client_api == "openai" and provider["wire"] == "anthropic"):
                        tiers[0].append(entry)
                    tiers[1].append(entry)
                tiers[2].append(entry)
        for tier in tiers:
            if tier:
                tier.sort(key=lambda e: (-e[2], self.calls.get(e[0]["id"], 0), e[0]["id"], e[1]["id"]))
                return tier
        return []

    def note_attempt(self, provider_id):
        self.calls[provider_id] = self.calls.get(provider_id, 0) + 1

    def note_failure(self, provider_id, failure):
        """Mark a provider dead when its refusal is permanent for this trial."""
        message = failure.message or ""
        if "budget is spent" in message or message.endswith(" is out"):
            self.dead.add(provider_id)


# ---------------------------------------------------------------- the server


class State:
    def __init__(self, env_url, directory, ratings, workload, probes, catalog):
        self.env_url = env_url
        self.directory = directory
        self.ratings = ratings
        self.workload = workload
        self.probes = probes
        self.catalog = catalog
        self.router = Router(probes, ratings)
        self.manifest_index = 0
        self.lock = threading.Lock()

    def requirements_for(self, req, client_api):
        """The manifest's row by arrival order, else a guess off the body."""
        manifest = self.workload["manifest"] if self.workload is not None else []
        index = self.manifest_index
        self.manifest_index += 1
        if index < len(manifest):
            return manifest[index]["requires"]
        vision = any(message["images"] for message in req.messages)
        return {
            "tools": len(req.tools) > 0,
            "vision": vision,
            "json": req.json_mode if client_api == "openai" else False,
        }

    def serve(self, client_api, body):
        """One client request: (status, content-type, payload bytes)."""
        if client_api == "openai":
            req = parse_openai_body(body)
        else:
            req = parse_anthropic_body(body)
        requires = self.requirements_for(req, client_api)
        candidates = self.router.candidates(requires, client_api, req.json_mode)
        for provider, model, _rating in candidates:
            self.router.note_attempt(provider["id"])
            try:
                answer = call_provider(self.env_url, provider, req, model["id"])
            except ProviderFailure as err:
                log("%s/%s refused (%s): %s" % (provider["id"], model["id"], err.status, err.message))
                self.router.note_failure(provider["id"], err)
                continue
            if client_api == "openai":
                if req.stream:
                    return 200, "text/event-stream", render_openai_sse(answer, model["id"]).encode("utf-8")
                payload = render_openai_json(answer, model["id"])
            else:
                if req.stream:
                    return 200, "text/event-stream", render_anthropic_sse(answer, model["id"]).encode("utf-8")
                payload = render_anthropic_json(answer, model["id"])
            return 200, "application/json", json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if client_api == "openai":
            payload = openai_error("no free model on the directory meets this request",
                                   "no_free_model", None)
        else:
            payload = anthropic_error("overloaded_error", "no free model on the directory meets this request")
        return 503, "application/json", json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def models_list(self, anthropic_shape):
        """GET /v1/models, in the shape the auth header names."""
        models = served_models(self.probes, self.ratings)
        seen = set()
        unique = []
        for model in models:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            unique.append(model)
        if anthropic_shape:
            return {"data": [{
                "type": "model", "id": m["id"], "display_name": m["display_name"],
                "created_at": FIXED_CREATED_AT,
            } for m in unique], "has_more": False}
        return {"object": "list", "data": [{
            "id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"],
        } for m in unique]}


# ---------------------------------------------------------------- HTTP plumbing


def _json_bytes(payload):
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    state = None  # set by main()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return value if _is_record(value) else None

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_error(self, wire, status, type_, message):
        payload = _json_bytes(
            anthropic_error(type_, message) if wire == "anthropic"
            else openai_error(message, type_, None))
        self._send(status, "application/json", payload)

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/v1/models":
            self._send_error("openai", 404, "not_found", "no route GET %s" % self.path)
            return
        with self.state.lock:
            payload = _json_bytes(self.state.models_list(
                isinstance(self.headers.get("x-api-key"), str)))
        self._send(200, "application/json", payload)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire, api = "openai", "openai"
        elif path == "/v1/messages":
            wire, api = "anthropic", "anthropic"
        else:
            self._send_error("openai", 404, "not_found", "no route POST %s" % path)
            return
        body = self._read_body()
        if body is None:
            self._send_error(wire, 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        try:
            with self.state.lock:
                status, content_type, payload = self.state.serve(api, body)
        except Exception as err:  # never leave the client hanging
            log("serving %s failed: %s" % (path, err))
            self._send_error(wire, 500, "internal_error", "the server failed to answer")
            return
        self._send(status, content_type, payload)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def register(env_url, port, catalog):
    """POST /register with the port and the one catalog; retry briefly."""
    url = env_url.rstrip("/") + "/register"
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    last = None
    for _attempt in range(5):
        try:
            status, raw = _exchange("POST", url, {"content-type": "application/json"},
                                    payload, REGISTER_TIMEOUT_S)
        except ProviderFailure as err:
            last = err
            threading.Event().wait(1.0)
            continue
        if status < 200 or status >= 300:
            raise RuntimeError("POST %s answered HTTP %s: %s" % (url, status, raw[:500]))
        try:
            reply = json.loads(raw.decode("utf-8"))
        except ValueError:
            reply = {}
        if isinstance(reply.get("requests"), int):
            log("registered: the workload holds %d requests" % reply["requests"])
        return
    raise RuntimeError("POST %s failed: %s" % (url, last))


def poll_work(env_url, server):
    """Poll GET /work every 100 ms; stop the server when it says done."""
    url = env_url.rstrip("/") + "/work"
    while True:
        try:
            status, raw = _exchange("GET", url, {}, None, WORK_TIMEOUT_S)
            if status == 200:
                work = json.loads(raw.decode("utf-8"))
                if _is_record(work) and work.get("done") is True:
                    server.shutdown()
                    return
        except (ProviderFailure, ValueError):
            pass
        threading.Event().wait(POLL_INTERVAL_S)


def load(env_url, prompt_file):
    """The brief, the probes and the one catalog (the --list document)."""
    directory, ratings, workload, error = read_brief(prompt_file)
    if error is not None:
        log(error)
    probes = []
    if env_url:
        for provider in directory["providers"]:
            probes.append(probe_provider(env_url, provider))
        for probe in probes:
            if not probe["ok"] and probe["error"] is not None:
                log(probe["error"])
    return State(env_url, directory, ratings, workload, probes,
                 build_catalog(probes, ratings))


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    state = load(env_url, prompt_file)
    if "--list" in argv:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(json.dumps(state.catalog, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0
    if env_url == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = Server(("0.0.0.0", 0), Handler)
    Handler.state = state
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models" % (
        port, sum(1 for p in state.probes if p["ok"]),
        len(served_models(state.probes, state.ratings))))
    try:
        register(env_url, port, state.catalog)
    except (RuntimeError, ProviderFailure) as err:
        log(str(err))
        return 1
    poller = threading.Thread(target=poll_work, args=(env_url, server), daemon=True)
    poller.start()

    def _stop(_signum, _frame):
        server.shutdown()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
