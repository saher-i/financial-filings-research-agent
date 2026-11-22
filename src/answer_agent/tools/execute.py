"""Execute requested evidence tools and attach inspectable source IDs."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .pages import expand_table_preview, read_pages
from .calculator import calculate


def _cache_key(name: str, arguments: dict) -> str:
    normalized = dict(arguments)
    if name == "calculate" and normalized.get("operation") == "grouped_change":
        normalized.pop("operands", None)  # grouped_change reads the complete source rows.
    return name + ":" + json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _previous_result(name: str, arguments: dict, trace: list[dict], messages: list[dict]):
    key = _cache_key(name, arguments)
    by_call = {message.get("tool_call_id"): message.get("content")
               for message in messages if message.get("role") == "tool"}
    for entry in reversed(trace):
        if (entry.get("tool") == name and "error" not in entry
                and isinstance(entry.get("arguments"), dict)
                and _cache_key(name, entry["arguments"]) == key
                and entry.get("call_id") in by_call):
            return entry, by_call[entry["call_id"]]
    return None


def _independent_result(name: str, arguments: dict, sql_tool, pdf_search):
    """Fetch evidence without touching the shared source list or conversation."""
    started = time.perf_counter()
    try:
        if name == "query_financials":
            result = sql_tool(arguments["sql"])
        else:
            result = pdf_search(arguments["query"], ticker=arguments.get("ticker"),
                                fiscal_year=arguments.get("fiscal_year"))
        return result, None, round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:
        return None, exc, round((time.perf_counter() - started) * 1000, 1)


def _prefetchable(call: dict) -> tuple[str, dict] | None:
    """Validate independent calls before starting work on another thread."""
    name = call.get("function", {}).get("name")
    if name not in {"query_financials", "search_filings"}:
        return None
    try:
        arguments = json.loads(call.get("function", {}).get("arguments", "{}"))
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, dict):
        return None
    if name == "query_financials":
        if not isinstance(arguments.get("sql"), str):
            return None
    else:
        if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
            return None
        if arguments.get("ticker") is not None and arguments["ticker"] not in {"AAPL", "MSFT", "GOOGL"}:
            return None
        if arguments.get("fiscal_year") is not None and arguments["fiscal_year"] not in {2024, 2025}:
            return None
    return name, arguments


def _execute_ordered_calls(state: dict, sql_tool, pdf_search, max_sources: int,
                           progress: Callable[[dict], None] | None = None,
                           page_reader=None, max_tool_calls: int = 24,
                           prefetched: dict[int, object] | None = None) -> dict:
    if type(max_tool_calls) is not int or max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    page_reader = page_reader or read_pages
    sources = list(state["sources"])
    messages = list(state["messages"])
    trace = list(state["trace"])
    known = {source["key"]: source["id"] for source in sources}
    for index, call in enumerate(state["pending_calls"]):
        name = call.get("function", {}).get("name", "")
        call_id = call.get("id", "unknown")
        stage = {"query_financials": "sql", "search_filings": "filings", "calculate": "calculation",
                 "read_pages": "pages"}.get(name, "tool")
        label = {"query_financials": "Checking financial figures",
                 "search_filings": "Searching annual filings",
                 "read_pages": "Reading original filing page",
                 "calculate": "Calculating from verified figures"}.get(name, "Running evidence tool")
        timed = progress is not None and name in {"query_financials", "search_filings", "read_pages", "calculate"}
        event_id = f"tool-{state.get('steps', 0)}-{call_id}"
        future = (prefetched or {}).get(index)
        started = time.perf_counter()
        duration_ms = None
        if timed and future is None:
            progress({"id": event_id, "phase": "start", "stage": stage, "message": label})
        failed = True
        source_count = len(sources)
        try:
            if len(trace) >= max_tool_calls:
                raise ValueError("Tool call limit reached; finish with existing evidence")
            arguments = json.loads(call.get("function", {}).get("arguments", "{}"))
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            previous = _previous_result(name, arguments, trace, messages)
            if previous is not None:
                entry, content = previous
                reused = {"tool": name, "call_id": call_id, "arguments": arguments,
                          "source_ids": entry.get("source_ids", []), "cached": True}
                if "result" in entry:
                    reused["result"] = entry["result"]
                trace.append(reused)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
                failed = False
                continue
            if name == "query_financials":
                sql = arguments["sql"]
                if not isinstance(sql, str):
                    raise ValueError("SQL must be text")
                if future is None:
                    result = sql_tool(sql)
                else:
                    result, error, duration_ms = future.result()
                    if error is not None:
                        raise error
                key = "sql:" + result["sql"]
                if key not in known:
                    if len(sources) >= max_sources:
                        raise ValueError("Evidence limit reached; answer from existing sources")
                    identifier = f"S{len(sources) + 1}"
                    source = {"id": identifier, "type": "sql", "key": key,
                              "sql": result["sql"], "columns": result["columns"],
                              "rows": result["rows"], "truncated": result.get("truncated", False),
                              "database": result.get("source", {}).get("database", "financials.db")}
                    sources.append(source)
                    known[key] = identifier
                output = {"source_id": known[key], "columns": result["columns"],
                          "rows": result["rows"], "truncated": result.get("truncated", False)}
                used = [known[key]]
            elif name == "search_filings":
                query = arguments["query"]
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("Search query must be text")
                ticker = arguments.get("ticker")
                fiscal_year = arguments.get("fiscal_year")
                if ticker is not None and ticker not in {"AAPL", "MSFT", "GOOGL"}:
                    raise ValueError("Unknown ticker")
                if fiscal_year is not None and fiscal_year not in {2024, 2025}:
                    raise ValueError("Unknown filing year")
                if future is None:
                    search = pdf_search(query, ticker=ticker, fiscal_year=fiscal_year)
                else:
                    search, error, duration_ms = future.result()
                    if error is not None:
                        raise error
                hits = search["results"] if isinstance(search, dict) else search
                output = {"passages": []}
                rewrite = search.get("rewrite") if isinstance(search, dict) else None
                used = []
                expanded_pages = []
                for hit in hits[:5]:
                    key = "pdf:" + hit["id"]
                    text, expanded = expand_table_preview(hit, page_reader)
                    if key not in known:
                        if len(sources) >= max_sources:
                            break
                        identifier = f"S{len(sources) + 1}"
                        sources.append({"id": identifier, "type": "pdf", "key": key,
                                        "chunk_id": hit["id"], "ticker": hit["ticker"],
                                        "fiscal_year": hit["fiscal_year"],
                                        "filename": hit["filename"], "pages": hit["pages"],
                                        "text": text, "page_expanded": expanded})
                        known[key] = identifier
                    identifier = known[key]
                    used.append(identifier)
                    if expanded:
                        expanded_pages.append({"source_id": identifier, "filename": hit["filename"],
                                               "page": hit["pages"][0]})
                    output["passages"].append({"source_id": identifier,
                                               "filename": hit["filename"], "pages": hit["pages"],
                                               "text": text[:1_800], "page_expanded": expanded})
            elif name == "calculate":
                output = calculate(sources, arguments)
                used = output["source_ids"]
            elif name == "read_pages":
                page_result = page_reader(arguments["filename"], arguments["page"],
                                          arguments.get("end_page"))
                key = "page:" + page_result["filename"] + ":" + ",".join(
                    str(page) for page in page_result["pages"])
                if key not in known:
                    if len(sources) >= max_sources:
                        raise ValueError("Evidence limit reached; answer from existing sources")
                    identifier = f"S{len(sources) + 1}"
                    filename = page_result["filename"]
                    sources.append({"id": identifier, "type": "pdf", "key": key,
                                    "ticker": filename.split("_", 1)[0],
                                    "fiscal_year": int(filename.split("FY", 1)[1][:4]),
                                    "filename": filename, "pages": page_result["pages"],
                                    "text": page_result["text"],
                                    "truncated": page_result.get("truncated", False)})
                    known[key] = identifier
                used = [known[key]]
                output = {"source_id": known[key], "filename": page_result["filename"],
                          "pages": page_result["pages"], "text": page_result["text"],
                          "truncated": page_result.get("truncated", False)}
            else:
                raise ValueError("Unknown tool")
            content = json.dumps(output, ensure_ascii=False, default=str)
            if len(content) > 20_000:
                raise ValueError("Tool output is too large; use a narrower query")
            entry = {"tool": name, "call_id": call_id, "arguments": arguments, "source_ids": used}
            if name == "search_filings" and rewrite:
                entry["rewrite"] = rewrite
            if name == "search_filings" and expanded_pages:
                entry["expanded_pages"] = expanded_pages
            if name == "calculate":
                entry["result"] = output
            trace.append(entry)
            failed = False
        except (KeyError, TypeError, ValueError, FileNotFoundError, TimeoutError, RuntimeError) as exc:
            # A rejected oversized output must not leave an uncited source behind.
            del sources[source_count:]
            known = {source["key"]: source["id"] for source in sources}
            content = json.dumps({"error": str(exc)}, ensure_ascii=False)
            trace.append({"tool": name, "error": str(exc)})
        finally:
            if timed:
                progress({"id": event_id, "phase": "complete", "stage": stage,
                          "message": label,
                          "duration_ms": (duration_ms if duration_ms is not None else
                                          round((time.perf_counter() - started) * 1000, 1)),
                          "status": "error" if failed else "ok"})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return {"messages": messages, "sources": sources, "trace": trace, "pending_calls": []}


def _execute_calls(state: dict, sql_tool, pdf_search, max_sources: int,
                   progress: Callable[[dict], None] | None = None,
                   page_reader=None, max_tool_calls: int = 24,
                   max_parallel_calls: int = 3) -> dict:
    """Overlap independent retrieval calls; commit results in model tool-call order."""
    if type(max_parallel_calls) is not int or not 1 <= max_parallel_calls <= 8:
        raise ValueError("max_parallel_calls must be between 1 and 8")
    pending = state["pending_calls"]
    current = state
    index = 0
    with ThreadPoolExecutor(max_workers=max_parallel_calls) as pool:
        while index < len(pending):
            end = index
            while end < len(pending) and _prefetchable(pending[end]) is not None:
                end += 1
            if end == index:
                end = index + 1
            batch = pending[index:end]
            prefetched = {}
            if len(batch) > 1 and max_parallel_calls > 1:
                previous = {}
                for offset, call in enumerate(batch):
                    if len(current["trace"]) + offset >= max_tool_calls:
                        break
                    prepared = _prefetchable(call)
                    if prepared is None:
                        continue
                    name, arguments = prepared
                    if _previous_result(name, arguments, current["trace"], current["messages"]):
                        continue
                    key = _cache_key(name, arguments)
                    future = previous.get(key)
                    if future is None:
                        future = pool.submit(_independent_result, name, arguments, sql_tool, pdf_search)
                        previous[key] = future
                    prefetched[offset] = future
                    if progress:
                        stage = "sql" if name == "query_financials" else "filings"
                        label = "Checking financial figures" if name == "query_financials" else "Searching annual filings"
                        progress({"id": f"tool-{current.get('steps', 0)}-{call.get('id', 'unknown')}",
                                  "phase": "start", "stage": stage, "message": label})
            current = _execute_ordered_calls(
                {**current, "pending_calls": batch}, sql_tool, pdf_search, max_sources,
                progress, page_reader, max_tool_calls, prefetched)
            index = end
    return current


