"""Gemini provider adapter implementing BaseProvider."""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import uuid

from core.models import Message, StreamEvent, StreamEventType, ToolCall, UsageMetadata, collapse_repeating_text
from core.retry import execute_stream_with_retry, execute_with_retry
from providers.base import BaseProvider, extract_and_resolve_images


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini models via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-flash-latest",
        client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        self._client = client

    @property
    def provider_name(self) -> str:
        return "gemini"

    @staticmethod
    def is_enterprise_mode() -> bool:
        """Determines whether Gemini Enterprise / Vertex AI mode is active."""
        import os
        if os.getenv("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1", "yes"):
            return True
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes"):
            return True
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            return True
        gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if gac and os.path.isfile(gac):
            return True
        default_sa = os.path.expanduser("~/.valstorm/gcp/valstorm-gemini-enterprise-sa.json")
        if os.path.isfile(default_sa):
            return True
        return False

    def _get_client(self) -> Any:
        """Get or initialize the Google GenAI async client."""
        if self._client is not None:
            return self._client
        import os
        from google import genai

        if self.is_enterprise_mode():
            creds = None
            gac_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if gac_json:
                try:
                    import json
                    from google.oauth2.service_account import Credentials
                    info = json.loads(gac_json)
                    creds = Credentials.from_service_account_info(
                        info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error("Failed to load GOOGLE_APPLICATION_CREDENTIALS_JSON: %s", e)

            if not creds and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                default_sa = os.path.expanduser("~/.valstorm/gcp/valstorm-gemini-enterprise-sa.json")
                if os.path.isfile(default_sa):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_sa

            project = (
                os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("VALSTORM_GCP_PROJECT")
                or "valstorm-gemini"
            )
            location = (
                os.getenv("GOOGLE_CLOUD_LOCATION")
                or os.getenv("VALSTORM_GCP_LOCATION")
                or "global"
            )
            client_kwargs = {
                "vertexai": True,
                "project": project,
                "location": location,
            }
            if creds:
                client_kwargs["credentials"] = creds
            self._client = genai.Client(**client_kwargs)
        else:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        """Convert standard or OpenAI-format tool schemas into Gemini FunctionDeclarations."""
        if not tools:
            return None

        from google.genai import types

        function_declarations: List[types.FunctionDeclaration] = []
        for t in tools:
            # Handle OpenAI wrapper format {"type": "function", "function": {...}}
            if "function" in t and isinstance(t["function"], dict):
                f_data = t["function"]
            else:
                f_data = t

            name = f_data.get("name", "")
            description = f_data.get("description", "")
            parameters = f_data.get("parameters")

            # Clean parameters schema if needed
            decl_kwargs: Dict[str, Any] = {
                "name": name,
                "description": description,
            }
            if parameters:
                decl_kwargs["parameters"] = parameters

            function_declarations.append(types.FunctionDeclaration(**decl_kwargs))

        return [types.Tool(function_declarations=function_declarations)]

    def _format_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Any]]:
        """Convert unified Message list into system_instruction and Gemini Content objects."""
        import json
        from google.genai import types

        system_parts: List[str] = []
        for msg in messages:
            if msg.role == "system" and msg.content:
                system_parts.append(msg.content.strip())
        system_instruction = "\n\n".join(system_parts) if system_parts else None

        raw_contents: List[types.Content] = []
        unsigned_call_ids: set = set()
        unsigned_tool_names: set = set()

        for msg in messages:
            if msg.role == "system":
                continue

            parts: List[types.Part] = []

            if msg.role == "user":
                if msg.content:
                    cleaned_content = collapse_repeating_text(msg.content)
                    if cleaned_content:
                        parts.append(types.Part(text=cleaned_content))
                # Multimodal image parts
                images = extract_and_resolve_images(msg)
                for img_bytes, mime_type in images:
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                if not parts:
                    parts.append(types.Part(text="..."))
                raw_contents.append(types.Content(role="user", parts=parts))

            elif msg.role in ("assistant", "model"):
                if msg.content:
                    cleaned_content = collapse_repeating_text(msg.content)
                    if cleaned_content:
                        parts.append(types.Part(text=cleaned_content))
                if msg.tool_calls:
                    is_foreign_provider = bool(msg.provider and str(msg.provider).lower() not in ("gemini", "google"))
                    for tc in msg.tool_calls:
                        sig = getattr(tc, "thought_signature", None)
                        if sig is not None:
                            if isinstance(sig, str):
                                import base64
                                try:
                                    sig = base64.b64decode(sig)
                                except Exception:
                                    sig = sig.encode("utf-8")
                            # If this tool name or ID was previously marked unsigned, re-enable signed mode
                            if tc.id in unsigned_call_ids:
                                unsigned_call_ids.remove(tc.id)
                            if tc.name in unsigned_tool_names:
                                unsigned_tool_names.remove(tc.name)
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc.name,
                                        args=tc.arguments or {},
                                        id=tc.id,
                                    ),
                                    thought_signature=sig,
                                )
                            )
                        elif is_foreign_provider:
                            # Tool call originates from a non-Gemini model (e.g. DeepSeek, OpenAI, Claude failover).
                            # Gemini 3.x strictly rejects functionCall parts without cryptographic thought_signature.
                            # Format as clean text observation so Gemini understands the historical tool execution without 400 error.
                            args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments or {})
                            parts.append(types.Part(text=f"[Executed tool `{tc.name}` with arguments: {args_str}]"))
                            if tc.id:
                                unsigned_call_ids.add(tc.id)
                            if tc.name:
                                unsigned_tool_names.add(tc.name)
                        else:
                            # Native Gemini tool call or unspecified provider (e.g. unit tests or historical messages)
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc.name,
                                        args=tc.arguments or {},
                                        id=tc.id,
                                    )
                                )
                            )
                if not parts:
                    parts.append(types.Part(text="..."))
                raw_contents.append(types.Content(role="model", parts=parts))

            elif msg.role == "tool":
                # In Gemini, function response is sent with role="user"
                tool_name = (
                    msg.tool_result.name
                    if msg.tool_result
                    else (getattr(msg, "name", None) or "tool")
                )
                output_val = (
                    msg.tool_result.output
                    if msg.tool_result
                    else (msg.content or "")
                )
                call_id = msg.tool_result.call_id if msg.tool_result else getattr(msg, "tool_call_id", None)

                is_unsigned = (
                    (call_id and call_id in unsigned_call_ids)
                    or (tool_name and tool_name in unsigned_tool_names)
                )

                if is_unsigned:
                    output_str = json.dumps(output_val) if isinstance(output_val, (dict, list)) else str(output_val or "")
                    parts.append(types.Part(text=f"[Historical tool result {tool_name}]: {output_str}"))
                    raw_contents.append(types.Content(role="user", parts=parts))
                else:
                    response_dict = (
                        output_val
                        if isinstance(output_val, dict)
                        else {"result": output_val}
                    )
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response=response_dict,
                                id=call_id,
                            )
                        )
                    )
                    raw_contents.append(types.Content(role="user", parts=parts))

        if not raw_contents:
            return system_instruction, [types.Content(role="user", parts=[types.Part(text="Hello")])]

        # Step 1: Merge consecutive turns of identical roles
        merged: List[types.Content] = []
        for c in raw_contents:
            if merged and merged[-1].role == c.role:
                for p in c.parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        # Deduplicate identical text parts within the same model turn
                        existing_texts = [getattr(ep, "text", None) for ep in merged[-1].parts if getattr(ep, "text", None)]
                        if p_text in existing_texts:
                            continue
                    merged[-1].parts.append(p)
            else:
                merged.append(c)

        # Step 2: Ensure first turn is user
        if merged and merged[0].role != "user":
            merged.insert(0, types.Content(role="user", parts=[types.Part(text="[Session resumed]")]))

        # Step 3: Validate function call & response pairing
        sanitized: List[types.Content] = []
        for c in merged:
            if c.role == "user":
                new_parts: List[types.Part] = []
                prev_turn = sanitized[-1] if sanitized else None
                prev_fc_names = set()
                if prev_turn and prev_turn.role == "model":
                    for p in prev_turn.parts:
                        if getattr(p, "function_call", None):
                            prev_fc_names.add(p.function_call.name)

                for p in c.parts:
                    if getattr(p, "function_response", None):
                        fn_name = p.function_response.name
                        if fn_name in prev_fc_names:
                            new_parts.append(p)
                        else:
                            resp_data = p.function_response.response
                            resp_str = json.dumps(resp_data) if isinstance(resp_data, (dict, list)) else str(resp_data)
                            new_parts.append(types.Part(text=f"[Historical tool result {fn_name}]: {resp_str}"))
                    else:
                        new_parts.append(p)
                sanitized.append(types.Content(role="user", parts=new_parts))
            elif c.role == "model":
                sanitized.append(c)

        # Step 4: Re-merge after conversions and ensure starting with user
        final_contents: List[types.Content] = []
        for c in sanitized:
            if final_contents and final_contents[-1].role == c.role:
                for p in c.parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        existing_texts = [getattr(ep, "text", None) for ep in final_contents[-1].parts if getattr(ep, "text", None)]
                        if p_text in existing_texts:
                            continue
                    final_contents[-1].parts.append(p)
            else:
                final_contents.append(c)

        if final_contents and final_contents[0].role != "user":
            final_contents.insert(0, types.Content(role="user", parts=[types.Part(text="[Session resumed]")]))

        return system_instruction, final_contents

    def _parse_response(
        self, response: Any, model_name: str
    ) -> Tuple[Message, UsageMetadata]:
        """Extract content, function calls, and usage metadata from Gemini response."""
        # 1. Extract usage metadata
        usage = UsageMetadata()
        raw_usage = getattr(response, "usage_metadata", None)
        if raw_usage is not None:
            usage.prompt_tokens = getattr(raw_usage, "prompt_token_count", 0) or 0
            usage.completion_tokens = (
                getattr(raw_usage, "candidates_token_count", 0) or 0
            )
            usage.total_tokens = getattr(raw_usage, "total_token_count", 0) or (
                usage.prompt_tokens + usage.completion_tokens
            )
            usage.cached_tokens = getattr(raw_usage, "cached_content_token_count", None)

        # 2. Extract text and function calls
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []

        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            content_obj = getattr(candidate, "content", None)
            if content_obj and getattr(content_obj, "parts", None):
                for part in content_obj.parts:
                    # Text part
                    p_text = getattr(part, "text", None)
                    if p_text:
                        text_parts.append(p_text)
                    # Function call part
                    p_fc = getattr(part, "function_call", None)
                    if p_fc:
                        fc_id = getattr(p_fc, "id", None) or str(uuid.uuid4())
                        fc_name = getattr(p_fc, "name", "")
                        fc_args = getattr(p_fc, "args", {})
                        p_sig = getattr(part, "thought_signature", None)
                        tool_calls.append(
                            ToolCall(
                                id=fc_id,
                                name=fc_name,
                                arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                                thought_signature=p_sig,
                            )
                        )
        elif hasattr(response, "text") and response.text:
            text_parts.append(response.text)

        # Fallback if raw function_calls helper populated but no parts
        if not tool_calls and getattr(response, "function_calls", None):
            for fc in response.function_calls:
                fc_id = getattr(fc, "id", None) or str(uuid.uuid4())
                fc_name = getattr(fc, "name", "")
                fc_args = getattr(fc, "args", {})
                tool_calls.append(
                    ToolCall(
                        id=fc_id,
                        name=fc_name,
                        arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                    )
                )

        content = "".join(text_parts).strip() if text_parts else None
        if content:
            content = collapse_repeating_text(content)

        msg = Message(
            role="assistant",
            content=content,
            model=model_name,
            provider="gemini",
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
        )
        return msg, usage

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[StreamEvent, Tuple[Message, UsageMetadata]]]:
        """Stream response chunks from Gemini, yielding StreamEvents and final (Message, UsageMetadata) with retry."""
        from google.genai import types

        model_name = model or self.default_model or "gemini-flash-latest"
        client = self._get_client()

        system_instruction, contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)

        config_args: Dict[str, Any] = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if gemini_tools:
            config_args["tools"] = gemini_tools

        for k in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if k in kwargs:
                config_args[k] = kwargs[k]

        config = types.GenerateContentConfig(**config_args) if config_args else None

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _stream_call():
            stream = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )

            all_text_parts: List[str] = []
            collected_tool_calls: List[ToolCall] = []
            last_usage_metadata = None

            async for chunk in stream:
                if chunk.usage_metadata:
                    last_usage_metadata = chunk.usage_metadata

                candidates = getattr(chunk, "candidates", None)
                if candidates and len(candidates) > 0:
                    content_obj = getattr(candidates[0], "content", None)
                    if content_obj and getattr(content_obj, "parts", None):
                        for part in content_obj.parts:
                            p_text = getattr(part, "text", None)
                            if p_text:
                                all_text_parts.append(p_text)
                                yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=p_text)

                            p_fc = getattr(part, "function_call", None)
                            if p_fc:
                                fc_id = getattr(p_fc, "id", None) or str(uuid.uuid4())
                                fc_name = getattr(p_fc, "name", "")
                                fc_args = getattr(p_fc, "args", {})
                                p_sig = getattr(part, "thought_signature", None)
                                tc = ToolCall(
                                    id=fc_id,
                                    name=fc_name,
                                    arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                                    thought_signature=p_sig,
                                )
                                collected_tool_calls.append(tc)
                                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)

                elif hasattr(chunk, "text") and chunk.text:
                    all_text_parts.append(chunk.text)
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=chunk.text)

            # Build usage metadata
            usage = UsageMetadata()
            if last_usage_metadata is not None:
                usage.prompt_tokens = getattr(last_usage_metadata, "prompt_token_count", 0) or 0
                usage.completion_tokens = getattr(last_usage_metadata, "candidates_token_count", 0) or 0
                usage.total_tokens = getattr(last_usage_metadata, "total_token_count", 0) or (
                    usage.prompt_tokens + usage.completion_tokens
                )
                usage.cached_tokens = getattr(last_usage_metadata, "cached_content_token_count", None)

            final_content = "".join(all_text_parts).strip() if all_text_parts else None
            if final_content:
                final_content = collapse_repeating_text(final_content)
            msg = Message(
                role="assistant",
                content=final_content,
                model=model_name,
                provider="gemini",
                tool_calls=collected_tool_calls if collected_tool_calls else None,
                usage=usage,
            )

            yield (msg, usage)

        async for item in execute_stream_with_retry(
            _stream_call,
            provider_name=self.provider_name,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        ):
            yield item

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        """Generate response non-streaming from Gemini with exponential backoff retry."""
        from google.genai import types

        model_name = model or self.default_model or "gemini-flash-latest"
        client = self._get_client()

        system_instruction, contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)

        config_args: Dict[str, Any] = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if gemini_tools:
            config_args["tools"] = gemini_tools

        for k in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if k in kwargs:
                config_args[k] = kwargs[k]

        config = types.GenerateContentConfig(**config_args) if config_args else None

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _call():
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return self._parse_response(response, model_name)

        return await execute_with_retry(
            _call,
            provider_name=self.provider_name,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        )
