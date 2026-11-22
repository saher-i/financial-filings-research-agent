"""Bounded LangGraph loop that gathers SQL/PDF evidence before answering."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, TypedDict

from src.config import load_local_fireworks_key
from src.financial_sql import get_schema

from .prompts import system_prompt
from .tools import TOOLS
from .tools.execute import _execute_calls
from .tools.filings import _pdf_search
from .tools.financial import query_financials
from .provider import FireworksChatProvider
from .planning import PLAN_TOOL, ResearchPlan, validate_research_plan
from .checks import (_format_calculated_precision, _has_unverified_numeric_claim, _normalize_citations,
                     _unverified_numeric_claims, _valid_source_ids)
from .verification import review_answer, revise_answer


class AgentState(TypedDict):
    question: str
    prior_messages: list[dict]
    messages: list[dict]
    sources: list[dict]
    trace: list[dict]
    pending_calls: list[dict]
    draft: str
    answer: str
    answer_status: str
    steps: int
    needs_revision: bool
    model_diagnostics: list[dict]
    failure_reason: str | None
    research_plan: ResearchPlan | None
    plan_status: str
    recovery_active: bool


def _finish(state: AgentState, max_steps: int) -> dict:
    draft = state["draft"].strip()
    sources = state["sources"]
    if state["pending_calls"] and state["steps"] >= max_steps:
        answer = "I reached the research step limit before I could verify a complete answer."
        status = "step_limit"
    elif not sources:
        answer = "I could not verify an answer from the supplied financial data or filings."
        status = "no_evidence"
    elif not draft:
        answer = "I found evidence but could not produce a supported answer."
        status = "no_draft"
    elif not _valid_source_ids(draft, sources):
        answer = "I found evidence but could not produce an answer with valid source citations."
        status = "citation_failed"
    elif _unverified_numeric_claims(draft, sources, state.get("trace", [])):
        answer = "I found financial evidence but could not verify a derived amount in the answer."
        status = "calculation_failed"
    else:
        answer = draft
        status = "answered"
    reason = state.get("failure_reason")
    if status == "no_draft" and not reason:
        reason = "empty_model_response_after_retry"
    elif status == "citation_failed":
        reason = "citations_missing_or_unknown"
    elif status == "step_limit":
        reason = "research_step_limit"
    elif status == "calculation_failed":
        reason = "required_calculation_missing"
    return {"answer": answer, "answer_status": status, "failure_reason": reason if status != "answered" else None}


def _retry_messages(state: AgentState) -> list[dict]:
    """Give an empty-response retry bounded evidence, without the long tool history."""
    evidence = []
    sources = state["sources"]
    question_terms = set(re.findall(r"[a-z0-9]{4,}", state["question"].casefold()))
    cited = set(re.findall(r"S\d+", state.get("draft", "")))
    def priority(source: dict) -> tuple[int, int]:
        body = str(source.get("rows", [])) if source.get("type") == "sql" else str(source.get("text", ""))
        words = set(re.findall(r"[a-z0-9]{4,}", body.casefold()))
        return (int(source.get("id") in cited), len(question_terms & words))
    chosen = sorted(sources, key=priority, reverse=True)[:12]
    for source in chosen:
        identifier = source.get("id")
        if source.get("type") == "sql":
            body = str(source.get("rows", []))[:6500]
        else:
            body = str(source.get("text", ""))[:1800]
        evidence.append(f"[{identifier}] {body}")
    for entry in state["trace"]:
        if entry.get("tool") == "calculate" and "error" not in entry:
            evidence.append(f"Calculation: {str(entry.get('result', {}))[:1400]}")
    return [state["messages"][0], *state.get("prior_messages", []),
            {"role": "user", "content": state["question"]},
            {"role": "user", "content": "Answer from the gathered evidence below. Do not use tools. "
             "Cite the listed source IDs for supported claims; say when a part is unverified. "
             "Keep the answer readable and cover every part.\n\n" + "\n\n".join(evidence)[:18000]}]


def build_graph(provider, sql_tool, pdf_search, max_steps: int = 6, max_sources: int = 24,
                progress: Callable[[dict], None] | None = None, page_reader=None,
                max_tool_calls: int = 24):
    """Compile a fresh graph around per-request dependencies (no shared chat state)."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph from starter/requirements.txt before using the agent") from exc

    if not 1 <= max_steps <= 12 or not 1 <= max_sources <= 50 or not 1 <= max_tool_calls <= 100:
        raise ValueError("Invalid agent limits")

    def ask_model(state: AgentState) -> dict:
        final_turn = state["steps"] == max_steps - 1 or len(state["sources"]) >= max_sources
        recovering = bool(state.get("recovery_active"))
        stage = "recovery" if recovering and not final_turn else "planning" if state["steps"] == 0 else "answer" if final_turn else "review"
        label = ("Finding missing evidence" if recovering and not final_turn else
                 "Choosing sources" if state["steps"] == 0 else
                 "Writing the answer from gathered evidence" if final_turn else
                 "Reviewing evidence and drafting answer")
        event_id = f"model-{state['steps'] + 1}"
        started = time.perf_counter()
        if progress:
            progress({"id": event_id, "phase": "start", "stage": stage, "message": label})
        messages_for_model = state["messages"]
        if final_turn:
            messages_for_model = [*messages_for_model, {"role": "user", "content":
                                  "This is your final response. Use the evidence already gathered; "
                                  "do not call tools. Answer every supported part with exact source citations "
                                  "and say clearly which parts remain unverified."}]
        tools_for_model = [] if final_turn else [*TOOLS, PLAN_TOOL] if state["steps"] == 0 and not recovering else TOOLS
        failed = False
        try:
            message = provider.complete(messages_for_model, tools_for_model)
        except Exception:
            failed = True
            raise
        finally:
            if progress:
                progress({"id": event_id, "phase": "complete", "stage": stage,
                          "message": label, "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                          "status": "error" if failed else "ok"})
        if not isinstance(message, dict):
            raise ValueError("Chat provider returned an invalid message")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("Chat provider returned invalid tool calls")
        diagnostics = [*state.get("model_diagnostics", [])]
        diagnostics.append({"turn": state["steps"] + 1, "retry": False,
                            **(message.get("diagnostics") or {})})
        if not calls and not (message.get("content") or "").strip() and state["sources"]:
            retry_start = time.perf_counter()
            if progress:
                progress({"id": f"{event_id}-retry", "phase": "start", "stage": "answer",
                          "message": "Retrying answer from gathered evidence"})
            retry_failed = False
            try:
                try:
                    message = provider.complete(_retry_messages(state), [], max_tokens=4000)
                except TypeError as exc:
                    if "max_tokens" not in str(exc):
                        raise
                    message = provider.complete(_retry_messages(state), [])
            except Exception:
                retry_failed = True
                raise
            finally:
                if progress:
                    progress({"id": f"{event_id}-retry", "phase": "complete", "stage": "answer",
                              "message": "Retrying answer from gathered evidence",
                              "duration_ms": round((time.perf_counter() - retry_start) * 1000, 1),
                              "status": "error" if retry_failed else "ok"})
            diagnostics.append({"turn": state["steps"] + 1, "retry": True,
                                **(message.get("diagnostics") or {})})
            calls = message.get("tool_calls") or []
        draft = _normalize_citations(_format_calculated_precision(
            message.get("content") or "", state["trace"]))
        history_message = dict(message)
        history_message.pop("diagnostics", None)
        if not calls:
            history_message.pop("tool_calls", None)
            history_message["content"] = draft
        messages = [*state["messages"], history_message]
        needs_revision = False
        if not calls and state["sources"] and draft and state["steps"] + 1 < max_steps:
            requests = []
            plan = state.get("research_plan") or {}
            if plan.get("needs_pdf") and not any(source.get("type") == "pdf" for source in state["sources"]):
                requests.append("Your research plan suggests the question needs filing evidence. "
                                "Check the user's actual request. Search the relevant 10-K if needed, "
                                "or state why that evidence is unavailable.")
            unverified = _unverified_numeric_claims(draft, state["sources"], state["trace"])
            if unverified:
                requests.append("These displayed figures are not verified against their cited evidence "
                                "or a matching calculate result: " + ", ".join(unverified[:5]) + ". "
                                "For directly reported figures, retrieve and cite the original value. "
                                "Use calculate with sourced operands for derived figures, or omit them entirely. "
                                "Do not repeat rejected figures while describing an earlier draft or tool failure.")
            if not _valid_source_ids(draft, state["sources"]):
                available = ", ".join(f"[{source['id']}]" for source in state["sources"])
                requests.append("Revise your answer using exact citations such as " + available +
                                ". Cite only source IDs that support your claims.")
            if requests:
                messages.append({"role": "user", "content": " ".join(requests)})
                needs_revision = True
        return {"messages": messages, "pending_calls": calls,
                "draft": draft, "steps": state["steps"] + 1,
                "needs_revision": needs_revision, "model_diagnostics": diagnostics,
                "failure_reason": "empty_model_response_after_retry" if not calls and not draft else None}

    def run_tools(state: AgentState) -> dict:
        plan_calls = [call for call in state["pending_calls"]
                      if call.get("function", {}).get("name") == "submit_research_plan"]
        evidence_calls = [call for call in state["pending_calls"]
                          if call.get("function", {}).get("name") != "submit_research_plan"]
        messages = list(state["messages"])
        plan = state.get("research_plan")
        status = state.get("plan_status", "not_provided")
        for call in plan_calls:
            try:
                plan = validate_research_plan(call.get("function", {}).get("arguments", ""))
                status = "valid"
                content = {"status": "recorded", "research_plan": plan}
            except ValueError as exc:
                if plan is None:
                    status = "invalid"
                content = {"error": str(exc), "instruction": "Continue selecting evidence tools safely."}
            messages.append({"role": "tool", "tool_call_id": call.get("id", "unknown"),
                             "content": json.dumps(content, ensure_ascii=False)})
        diagnostics = [*state.get("model_diagnostics", [])]
        if plan_calls and diagnostics:
            diagnostics[-1] = {**diagnostics[-1], "plan_status": status,
                               "research_plan": plan}
        if evidence_calls:
            update = _execute_calls({**state, "messages": messages, "pending_calls": evidence_calls},
                                    sql_tool, pdf_search, max_sources, progress, page_reader,
                                    max_tool_calls)
        else:
            update = {"messages": messages, "pending_calls": []}
        return {**update, "research_plan": plan, "plan_status": status,
                "model_diagnostics": diagnostics}

    def next_step(state: AgentState) -> str:
        if state["pending_calls"] and state["steps"] < max_steps:
            return "tools"
        if state.get("needs_revision") and state["steps"] < max_steps:
            return "model"
        return "finish"

    workflow = StateGraph(AgentState)
    workflow.add_node("model", ask_model)
    workflow.add_node("tools", run_tools)
    workflow.add_node("finish", lambda state: _finish(state, max_steps))
    workflow.add_edge(START, "model")
    workflow.add_conditional_edges("model", next_step, {"tools": "tools", "model": "model", "finish": "finish"})
    workflow.add_edge("tools", "model")
    workflow.add_edge("finish", END)
    return workflow.compile()


def answer_question(question: str, *, strategy: str = "structure", model: str | None = None,
                    provider=None, sql_tool=None, pdf_search=None, schema: dict | None = None,
                    db_path: Path | None = None, max_steps: int = 8,
                    progress: Callable[[dict], None] | None = None,
                    page_reader=None, answer_style: str = "concise", verify: bool = False,
                    max_tool_calls: int = 24, history: list[dict] | None = None) -> dict:
    """Return an answer, inspectable evidence, and tool trace for one question."""
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("Question must be 1–4000 characters")
    if strategy not in {"structure", "semantic"}:
        raise ValueError("Unknown chunking strategy")
    if answer_style not in {"concise", "detailed"}:
        raise ValueError("Choose concise or detailed answer style")
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    prior = history if history is not None else []
    if (not isinstance(prior, list) or len(prior) > 8 or len(prior) % 2
            or any(not isinstance(item, dict) or item.get("role") != ("user" if index % 2 == 0 else "assistant")
                   or not isinstance(item.get("content"), str) or len(item["content"]) > 2000
                   for index, item in enumerate(prior))):
        raise ValueError("History must contain at most four bounded user and assistant exchanges")
    if provider is None or pdf_search is None:
        load_local_fireworks_key()
    provider = provider or FireworksChatProvider(model)
    sql_tool = sql_tool or (lambda sql: query_financials(sql, db_path=db_path))
    pdf_search = pdf_search or (lambda query, ticker=None, fiscal_year=None:
                                _pdf_search(query, strategy, ticker, fiscal_year))
    schema = schema or get_schema(db_path)
    graph = build_graph(provider, sql_tool, pdf_search, max_steps,
                        progress=progress, page_reader=page_reader,
                        max_tool_calls=max_tool_calls)
    prompt = system_prompt(schema, answer_style)
    if prior:
        prompt += ("\nPrior conversation is context for resolving follow-up references only. "
                   "It is not evidence. Recheck current claims with tools and cite only sources "
                   "gathered for this question. Do not treat old citation labels as current sources.")
    initial: AgentState = {"question": question, "prior_messages": prior, "messages": [
        {"role": "system", "content": prompt}, *prior,
        {"role": "user", "content": question}],
        "sources": [], "trace": [], "pending_calls": [], "draft": "", "answer": "",
        "answer_status": "", "steps": 0, "needs_revision": False,
        "model_diagnostics": [], "failure_reason": None,
        "research_plan": None, "plan_status": "not_provided", "recovery_active": False}
    result = graph.invoke(initial, config={"recursion_limit": max_steps * 3 + 3})
    verification = {"status": "not_run", "findings": [], "reason": None,
                    "method": "optional_model_review", "revision_attempted": False,
                    "review_calls": 0}
    if verify and result["answer_status"] == "answered":
        first = review_answer(question, result["answer"], result["sources"], provider, progress,
                              research_plan=result.get("research_plan"), trace=result["trace"])
        verification = {**first, "method": "model_assisted", "revision_attempted": False,
                        "review_calls": len(first.get("attempts", [])) or 1}
        result["trace"].append({"tool": "answer_review", "status": first["status"],
                                "findings": first["findings"],
                                "duration_ms": first["duration_ms"]})
        if first["status"] == "needs_revision" and first.get("evidence_gap"):
            gap = first["evidence_gap"]
            verification["recovery_attempted"] = True
            recovery_prompt = ("A review found this required part unsupported by the gathered evidence: "
                               f"{gap}. Make one targeted evidence request using an available tool. "
                               "Use the new evidence to answer the user's actual question. "
                               "Do not repeat the research plan or search unrelated topics. "
                               "If the evidence remains unavailable, state the limitation.")
            recovery_graph = build_graph(provider, sql_tool, pdf_search, max_steps=2,
                                         progress=progress, page_reader=page_reader,
                                         max_tool_calls=min(100, len(result["trace"]) + 3))
            recovered = recovery_graph.invoke({**result,
                "messages": [*result["messages"], {"role": "user", "content": recovery_prompt}],
                "steps": 0, "pending_calls": [], "draft": "", "needs_revision": False,
                "recovery_active": True}, config={"recursion_limit": 9})
            new_tools = recovered["trace"][len(result["trace"]):]
            verification["recovery_tool_calls"] = sum(
                entry.get("tool") in {"query_financials", "search_filings", "read_pages", "calculate"}
                and "error" not in entry for entry in new_tools)
            recovery_event = {"tool": "evidence_recovery", "request": gap,
                              "status": "new_evidence" if verification["recovery_tool_calls"]
                              else "no_new_evidence", "tool_calls": verification["recovery_tool_calls"]}
            if recovered["answer_status"] == "answered":
                result = recovered
                result["trace"].append(recovery_event)
                second = review_answer(question, result["answer"], result["sources"], provider,
                                       progress, research_plan=result.get("research_plan"),
                                       trace=result["trace"])
                verification["review_calls"] += len(second.get("attempts", [])) or 1
                verification["final_review"] = second
                result["trace"].append({"tool": "answer_review", "status": second["status"],
                                        "findings": second["findings"],
                                        "duration_ms": second["duration_ms"]})
                if second["status"] == "supported":
                    verification["status"] = "supported_after_recovery"
                    verification["findings"] = []
                else:
                    result["answer_status"] = "needs_verification"
                    verification["status"] = second["status"]
                    verification["findings"] = second["findings"]
            else:
                result["trace"].append(recovery_event)
                verification["reason"] = "targeted_recovery_did_not_resolve_gap"
                result["answer_status"] = "needs_verification"
            first = {**first, "status": "recovery_handled"}
        if first["status"] == "needs_revision":
            verification["revision_attempted"] = True
            revision, diagnostics = revise_answer(question, result["answer"], result["sources"],
                                                  first["findings"], provider, progress)
            result["trace"].append({"tool": "answer_revision", "status": "returned" if revision else "empty",
                                    "diagnostics": diagnostics})
            revision = _normalize_citations(_format_calculated_precision(revision, result["trace"]))
            if revision and _valid_source_ids(revision, result["sources"]):
                second = review_answer(question, revision, result["sources"], provider, progress,
                                       research_plan=result.get("research_plan"), trace=result["trace"])
                verification["review_calls"] += len(second.get("attempts", [])) or 1
                verification["final_review"] = second
                result["trace"].append({"tool": "answer_review", "status": second["status"],
                                        "findings": second["findings"],
                                        "duration_ms": second["duration_ms"]})
                if second["status"] == "supported":
                    result["answer"] = revision
                    verification["status"] = "supported_after_revision"
                    verification["findings"] = []
                else:
                    result["answer_status"] = "needs_verification"
            else:
                verification["reason"] = "revision_missing_or_invalid_citations"
                result["answer_status"] = "needs_verification"
        elif first["status"] not in {"supported", "recovery_handled"}:
            result["answer_status"] = "needs_verification"
    return {"answer": result["answer"], "answer_status": result["answer_status"],
            "draft": result["draft"], "sources": [
        {key: value for key, value in source.items() if key != "key"}
        for source in result["sources"]], "trace": result["trace"],
            "model": getattr(provider, "model", model or "injected"),
            "answer_style": answer_style,
            "verification": verification,
            "research_plan": result.get("research_plan"),
            "plan_status": result.get("plan_status", "not_provided"),
            "model_diagnostics": result.get("model_diagnostics", []),
            "failure_reason": result.get("failure_reason")}
