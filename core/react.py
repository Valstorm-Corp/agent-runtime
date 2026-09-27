"""ReAct (Reasoning and Acting) execution engine for multi-turn agent execution with streaming support."""

import asyncio
import inspect
import json
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from .models import Message, SessionState, StreamEvent, StreamEventType, ToolCall, ToolResult, UsageMetadata, collapse_repeating_text
from .sanitizer import sanitize_text
from .keystore import KeyStore
from .compaction import ContextCompactor

DEFAULT_MAX_ITERATIONS = int(os.environ.get("VALSTORM_MAX_ITERATIONS", "250"))

# --- Continuation guard -------------------------------------------------------------------
# Models (especially after backend changes) often end a turn by handing executable work back
# to the user ("please run the tests and let me know"). When the agent has a shell tool, we
# nudge it once to do that work itself. Disable with VALSTORM_CONTINUATION_GUARD=0.
CONTINUATION_GUARD_ENABLED = os.environ.get("VALSTORM_CONTINUATION_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
MAX_CONTINUATION_NUDGES = int(os.environ.get("VALSTORM_MAX_CONTINUATION_NUDGES", "2"))

_HANDOFF_PATTERN = re.compile(
    r"(?i)("
    r"please\s+(re-?run|run|execute|test|try|verify|restart|rebuild|install|apply|check)\b"
    r"|you\s+(can|should|will need to|need to|may need to|could)\s+(now\s+)?(re-?run|run|execute|test|verify|restart|rebuild|install)\b"
    r"|run\s+(the\s+following|this|these)\s+commands?"
    r"|let\s+me\s+know\s+(the|what|if\s+the|how\s+the|whether\s+the)\s+(result|output|tests?|command|build|error)"
    r"|once\s+you(\s+have|'ve)?\s+(run|executed|applied|installed|tested|restarted|rebuilt|provided)"
    r"|i\s+will\s+(now|next|then)\s+\w+"
    r"|(let\s+me|i'll|i\s+will)\s+(now\s+)?(run|execute|check|verify|test|apply|patch|fix|update|read|look)\b[^.?!]*[.:]?\s*$"
    r")"
)
_NO_EXEC_PATTERN = re.compile(r"(?i)(do\s*n[o']?t|dont|never)\s+(run|execute)|without\s+(running|executing)|no\s+commands|read-?only")

CONTINUATION_NUDGE = (
    "[System Notice: Your last message stopped before the work was finished — it describes a next step or asks "
    "the user to run/verify something. You have tools (e.g. terminal_exec, read_file, patch_file) and can do this "
    "yourself. Unless it genuinely requires something only the user can provide (credentials, a GUI/device action, "
    "a product decision) or the user told you not to run commands, do it now and keep iterating until the task is "
    "verified (tests/commands actually run and passing). If you truly cannot proceed, state exactly why in one line.]"
)
TRUNCATED_NUDGE = (
    "[System Notice: Your previous response was cut off by the output token limit. Continue exactly where you left "
    "off. If you were about to call a tool, call it now; if you were writing a large file, split it into smaller "
    "write_file/patch_file calls.]"
)
MALFORMED_CALL_NUDGE = (
    "[System Notice: Your previous tool call was malformed and could not be parsed. Retry the tool call with valid, "
    "complete JSON arguments. For very large content, split it across multiple smaller calls.]"
)


def _needs_continuation(text: str, tool_names: List[str], last_user_text: str) -> bool:
    """True when a final answer hands executable work back to the user although the agent could do it."""
    if not CONTINUATION_GUARD_ENABLED or not text:
        return False
    if not any(t in tool_names for t in ("terminal_exec", "execute_code", "patch_file", "write_file")):
        return False
    if last_user_text and _NO_EXEC_PATTERN.search(last_user_text):
        return False
    tail = text.strip()[-700:]
    return bool(_HANDOFF_PATTERN.search(tail))


class ReActEngine:
    """Orchestrates the ReAct execution loop (Thought -> Action -> Observation -> Final Answer)."""

    def __init__(
        self,
        provider: Any,
        tools: Any,
        keystore: Optional[KeyStore] = None,
        compactor: Optional[ContextCompactor] = None,
    ):
        self.provider = provider
        self.tools = tools
        self.keystore = keystore
        self.compactor = compactor or ContextCompactor()
        self.steer_queue: List[str] = []

    def steer(self, message: str) -> None:
        """Enqueues an out-of-band steering instruction to inject on the next loop iteration."""
        if message and message.strip():
            self.steer_queue.append(message.strip())

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Executes a single ToolCall against the tool registry safely with precision timing."""
        name = tool_call.name
        args = tool_call.arguments or {}
        start_t = time.perf_counter()

        try:
            if hasattr(self.tools, "execute_async"):
                res = await self.tools.execute_async(name, args, call_id=tool_call.id)
            elif hasattr(self.tools, "execute"):
                try:
                    if inspect.iscoroutinefunction(self.tools.execute):
                        res = await self.tools.execute(name, args, call_id=tool_call.id)
                    else:
                        res = await asyncio.to_thread(self.tools.execute, name, args, call_id=tool_call.id)
                    if inspect.isawaitable(res):
                        res = await res
                except TypeError:
                    try:
                        if inspect.iscoroutinefunction(self.tools.execute):
                            res = await self.tools.execute(name, args)
                        else:
                            res = await asyncio.to_thread(self.tools.execute, name, args)
                        if inspect.isawaitable(res):
                            res = await res
                    except TypeError:
                        if inspect.iscoroutinefunction(self.tools.execute):
                            res = await self.tools.execute(name, **args)
                        else:
                            res = await asyncio.to_thread(self.tools.execute, name, **args)
                        if inspect.isawaitable(res):
                            res = await res
            elif hasattr(self.tools, "get"):
                tool_fn = self.tools.get(name)
                if tool_fn is None:
                    raise ValueError(f"Tool '{name}' not found in registry.")
                if inspect.iscoroutinefunction(tool_fn):
                    res = await tool_fn(**args)
                else:
                    res = await asyncio.to_thread(tool_fn, **args)
                if inspect.isawaitable(res):
                    res = await res
            else:
                raise AttributeError("Tool registry does not have execute or get method.")

            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            if isinstance(res, ToolResult):
                if not res.call_id:
                    res.call_id = tool_call.id
                if not res.duration_ms:
                    res.duration_ms = dur_ms
                if res.output:
                    res.output = sanitize_text(res.output)
                return res

            out_str = json.dumps(res, indent=2) if isinstance(res, (dict, list)) else str(res)
            out_str = sanitize_text(out_str)
            return ToolResult(
                call_id=tool_call.id,
                name=name,
                output=out_str,
                is_error=False,
                duration_ms=dur_ms,
                payload_bytes=len(out_str.encode("utf-8")),
                item_count=len(res) if isinstance(res, (list, dict)) else None,
            )
        except Exception as e:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            err_msg = sanitize_text(f"Error executing tool '{name}': {str(e)}")
            return ToolResult(
                call_id=tool_call.id,
                name=name,
                output=err_msg,
                is_error=True,
                duration_ms=dur_ms,
                payload_bytes=len(err_msg.encode("utf-8")),
            )

    async def run_turn_stream(
        self,
        session: SessionState,
        user_input: str,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Executes a full ReAct loop turn with real-time streaming of tokens and tool execution events."""
        iter_limit = max_iterations if max_iterations is not None else DEFAULT_MAX_ITERATIONS
        target_model = model or session.active_model
        # Avoid duplicating user prompt if already hydrated in session
        if not session.messages or session.messages[-1].content != user_input or session.messages[-1].role != "user":
            user_msg = Message(role="user", content=user_input, images=images)
            session.add_message(user_msg)

        # 1. Pre-turn compaction check: clear context headroom before entering tool loop
        if self.compactor and self.compactor.should_auto_compact(session):
            c_res = self.compactor.compact(session)
            if c_res.compacted:
                yield StreamEvent(
                    event_type=StreamEventType.CONTEXT_COMPACTED,
                    delta=f"\n\033[93m🧹 [Auto-Compaction (Pre-Turn)]: Saved ~{c_res.tokens_saved:,} tokens ({c_res.compression_ratio:.1%} retained)\033[0m\n",
                    metadata={
                        "tokens_saved": c_res.tokens_saved,
                        "compression_ratio": c_res.compression_ratio,
                        "summary": c_res.summary,
                        "phase": "pre-turn",
                    },
                )

        assistant_msg: Optional[Message] = None
        usage_meta: Optional[UsageMetadata] = None
        empty_retries = 0
        max_empty_retries = 2
        continuation_nudges = 0
        truncation_retries = 0
        malformed_retries = 0
        turn_tool_calls = 0

        for iteration in range(iter_limit):
            # 2. In-loop compaction check: emergency safeguard for runaway tool results
            if self.compactor and self.compactor.should_auto_compact(session):
                c_res = self.compactor.compact(session)
                if c_res.compacted:
                    yield StreamEvent(
                        event_type=StreamEventType.CONTEXT_COMPACTED,
                        delta=f"\n\033[93m🧹 [Auto-Compaction]: Saved ~{c_res.tokens_saved:,} tokens ({c_res.compression_ratio:.1%} retained)\033[0m\n",
                        metadata={
                            "tokens_saved": c_res.tokens_saved,
                            "compression_ratio": c_res.compression_ratio,
                            "summary": c_res.summary,
                            "phase": "in-loop",
                        },
                    )

            # Inject any queued out-of-band steering instructions
            if self.steer_queue:
                steer_msgs = list(self.steer_queue)
                self.steer_queue.clear()
                combined_steer = "\n".join(steer_msgs)
                steer_msg_obj = Message(
                    role="user",
                    content=f"[OUT-OF-BAND USER STEERING INSTRUCTION]\n{combined_steer}\n[/OUT-OF-BAND USER STEERING INSTRUCTION]",
                    model=target_model,
                    provider=getattr(self.provider, "provider_name", "unknown"),
                )
                session.add_message(steer_msg_obj)
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_CHUNK,
                    delta=f"\n\033[96m🎯 [Steering Injected]: {combined_steer}\033[0m\n",
                )

            schemas = []
            if hasattr(self.tools, "get_schemas"):
                schemas = self.tools.get_schemas()
            elif hasattr(self.tools, "schemas"):
                schemas = self.tools.schemas

            assistant_msg = None
            usage_meta = None

            set_ctx = getattr(self.provider, "set_request_context", None)
            if callable(set_ctx):
                set_ctx(chat_id=(session.metadata or {}).get("valstorm_chat_id") or session.session_id)

            # Check if provider supports streaming
            if hasattr(self.provider, "generate_stream"):
                stream = self.provider.generate_stream(
                    messages=session.messages,
                    tools=schemas,
                    model=target_model,
                )
                async for item in stream:
                    if isinstance(item, StreamEvent):
                        if item.event_type == StreamEventType.TURN_COMPLETE:
                            # A provider's TURN_COMPLETE only ends this single LLM call. Re-label it so
                            # consumers don't treat a mid-turn call as the end of the whole turn.
                            if item.message:
                                assistant_msg = item.message
                                if item.usage and not usage_meta:
                                    usage_meta = item.usage
                            yield StreamEvent(
                                event_type=StreamEventType.LLM_CALL_COMPLETE,
                                message=item.message,
                                usage=item.usage,
                                metadata=item.metadata,
                            )
                            continue
                        yield item
                    elif isinstance(item, tuple) and len(item) >= 2:
                        assistant_msg, usage_meta = item[0], item[1]
                    elif isinstance(item, Message):
                        assistant_msg = item
            else:
                if inspect.iscoroutinefunction(self.provider.generate):
                    res = await self.provider.generate(
                        messages=session.messages,
                        tools=schemas,
                        model=target_model,
                    )
                else:
                    res = self.provider.generate(
                        messages=session.messages,
                        tools=schemas,
                        model=target_model,
                    )
                    if inspect.isawaitable(res):
                        res = await res

                if isinstance(res, tuple):
                    assistant_msg, usage_meta = res[0], res[1]
                else:
                    assistant_msg = res

                if assistant_msg and assistant_msg.content:
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=assistant_msg.content)
                if assistant_msg and assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)

            if not isinstance(assistant_msg, Message):
                raise TypeError(f"Provider must produce a Message instance, got {type(assistant_msg)}")

            if not assistant_msg.model and target_model:
                assistant_msg.model = target_model
            if not assistant_msg.provider and hasattr(self.provider, "provider_name"):
                assistant_msg.provider = getattr(self.provider, "provider_name")
            if usage_meta and not assistant_msg.usage:
                assistant_msg.usage = usage_meta

            # Add to session (aggregates tokens)
            session.add_message(assistant_msg)

            # Process tool calls (executed concurrently in parallel)
            if assistant_msg.tool_calls and len(assistant_msg.tool_calls) > 0:
                empty_retries = 0
                turn_tool_calls += len(assistant_msg.tool_calls)
                for tool_call in assistant_msg.tool_calls:
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_EXECUTION_START,
                        tool_call=tool_call,
                    )

                # Parallel tool execution via asyncio.gather
                tool_results = await asyncio.gather(
                    *[self._execute_tool(tc) for tc in assistant_msg.tool_calls]
                )

                for tool_call, tool_result in zip(assistant_msg.tool_calls, tool_results):
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_EXECUTION_RESULT,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )

                    if isinstance(tool_result.output, (str, int, float, bool)):
                        content_str = str(tool_result.output)
                    else:
                        try:
                            content_str = json.dumps(tool_result.output)
                        except Exception:
                            content_str = str(tool_result.output)

                    tool_msg = Message(
                        role="tool",
                        content=content_str,
                        tool_result=tool_result,
                        model=target_model,
                        provider=assistant_msg.provider,
                    )
                    session.add_message(tool_msg)
            else:
                has_content = bool(assistant_msg.content and assistant_msg.content.strip())
                finish = (assistant_msg.finish_reason or "").lower()
                can_continue = iteration < iter_limit - 1

                # Output was cut off by the token limit: ask the model to continue instead of ending.
                if finish in ("length", "max_tokens") and truncation_retries < 2 and can_continue:
                    truncation_retries += 1
                    session.add_message(Message(role="user", content=TRUNCATED_NUDGE, model=target_model, provider=assistant_msg.provider))
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="\n\033[93m↻ [Output truncated — asking the model to continue]\033[0m\n")
                    continue

                # Malformed function call: retry the call instead of ending the turn silently.
                if "malformed" in finish and malformed_retries < 2 and can_continue:
                    malformed_retries += 1
                    session.add_message(Message(role="user", content=MALFORMED_CALL_NUDGE, model=target_model, provider=assistant_msg.provider))
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="\n\033[93m↻ [Malformed tool call — retrying]\033[0m\n")
                    continue

                # Continuation guard: the model handed executable work back to the user.
                if has_content and continuation_nudges < MAX_CONTINUATION_NUDGES and can_continue:
                    tool_names: List[str] = []
                    try:
                        tool_names = [
                            (sc.get("function", {}) or {}).get("name") or sc.get("name", "")
                            for sc in (schemas or [])
                            if isinstance(sc, dict)
                        ]
                    except Exception:
                        tool_names = []
                    if _needs_continuation(assistant_msg.content or "", tool_names, user_input or ""):
                        continuation_nudges += 1
                        session.add_message(Message(role="user", content=CONTINUATION_NUDGE, model=target_model, provider=assistant_msg.provider))
                        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="\n\033[96m↻ [Continuing: agent handed work back — nudging it to do it itself]\033[0m\n")
                        continue

                if not has_content:
                    if empty_retries < max_empty_retries and (iteration < iter_limit - 1):
                        empty_retries += 1
                        nudge_msg = Message(
                            role="user",
                            content="[System Notice: The previous response was empty. Please provide your response or next tool call.]",
                            model=target_model,
                            provider=assistant_msg.provider,
                        )
                        session.add_message(nudge_msg)
                        continue
                    else:
                        fallback_text = "(Model concluded turn without text response)"
                        assistant_msg.content = fallback_text
                        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=f"\n\033[93m⚠️ {fallback_text}\033[0m\n")

                # No tool calls: final answer reached
                if assistant_msg and assistant_msg.content:
                    assistant_msg.content = collapse_repeating_text(assistant_msg.content)
                    assistant_msg.body = assistant_msg.content

                yield StreamEvent(
                    event_type=StreamEventType.TURN_COMPLETE,
                    message=assistant_msg,
                    usage=assistant_msg.usage,
                )
                return

        # Max iterations reached: yield limit notice event
        yield StreamEvent(
            event_type=StreamEventType.ITERATION_LIMIT_REACHED,
            delta=f"\n\033[93m⚠️ [Loop Bound]: Reached iteration limit ({iter_limit}).\033[0m\n",
            metadata={
                "iteration_count": iter_limit,
                "has_active_tool_calls": bool(assistant_msg and assistant_msg.tool_calls),
            },
        )

        if assistant_msg is not None:
            if assistant_msg.content:
                assistant_msg.content = collapse_repeating_text(assistant_msg.content)
                assistant_msg.body = assistant_msg.content
            yield StreamEvent(
                event_type=StreamEventType.TURN_COMPLETE,
                message=assistant_msg,
                usage=assistant_msg.usage,
            )
            return

        fallback_msg = Message(
            role="assistant",
            content=f"Terminated after reaching maximum iterations ({iter_limit}). Use --max-iterations or /continue to extend.",
            model=target_model,
            provider=getattr(self.provider, "provider_name", "unknown"),
        )
        session.add_message(fallback_msg)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=fallback_msg,
            usage=fallback_msg.usage,
        )

    async def run_turn(
        self,
        session: SessionState,
        user_input: str,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        step_callback: Optional[Callable[[str, Any], None]] = None,
    ) -> Message:
        """Executes a full ReAct loop turn (non-streaming wrapper around run_turn_stream)."""
        final_message = None

        async for event in self.run_turn_stream(
            session=session,
            user_input=user_input,
            model=model,
            max_iterations=max_iterations,
        ):
            if step_callback:
                if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                    step_callback("text_chunk", event.delta)
                elif event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                    step_callback("tool_call", event.tool_call)
                elif event.event_type == StreamEventType.TOOL_EXECUTION_RESULT and event.tool_result:
                    step_callback("tool_result", event.tool_result)

            if event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                final_message = event.message

        if final_message is not None:
            return final_message
        if session.messages and session.messages[-1].role == "assistant":
            return session.messages[-1]
        return Message(role="assistant", content="No response generated.")
