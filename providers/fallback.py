"""Multi-provider fallback cascade engine with automatic error recovery and failover.

Prioritizes Gemini ($0.00 credit pool), then DigitalOcean Serverless Inference
(DeepSeek, Kimi, GPT-oss-120b, Llama, Mistral under ONE payment gateway),
with direct proprietary providers as emergency safety nets.
"""

import asyncio
import inspect
import logging
import os
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from core.keystore import KeyStore
from core.models import Message, StreamEvent, StreamEventType, ToolCall, ToolResult, UsageMetadata
from core.retry import extract_status_code, is_retryable_error
from providers.base import BaseProvider

logger = logging.getLogger("vsagent.fallback")


def is_provider_failover_candidate(exc: BaseException) -> bool:
    """Determines whether an exception is a candidate for cascading to an alternative provider tier.

    Transient errors (503, 429, timeouts) and provider-specific errors (e.g. 400 thought_signature,
    401 expired key, 404 model removed) trigger failover to the next provider.
    Pure internal application bugs (ValueError, TypeError, etc.) are fatal and not cascaded.
    """
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return False

    if is_retryable_error(exc):
        return True

    code = extract_status_code(exc)
    if code is not None:
        # Any HTTP/API response from an external provider endpoint is a candidate for failover
        return True

    exc_str = (str(exc) + " " + repr(exc)).lower()
    if any(p in exc_str for p in ("thought_signature", "thought signature", "quota", "rate limit", "unavailable")):
        return True

    exc_type = type(exc).__name__
    if any(k in exc_type for k in ("API", "ClientError", "ServerError", "HTTP", "GenAI")):
        return True

    # Pure Python built-in exceptions indicate monorepo code or schema bugs; raise immediately
    if isinstance(exc, (ValueError, TypeError, AttributeError, KeyError, IndexError, AssertionError, NotImplementedError)):
        return False

    return False

# Default Tier 2 (Reasoner / Monorepo Dev / QA) Cascade Hierarchy
DEFAULT_REASONER_CASCADE_SPECS = [
    {"provider": "gemini", "model": "gemini-flash-latest", "label": "Gemini (Primary Credits)"},
    {"provider": "do", "model": "deepseek-v4-pro", "label": "DO DeepSeek V4 Pro"},
    {"provider": "do", "model": "kimi-k2.6", "label": "DO Kimi K2.6"},
    {"provider": "do", "model": "openai/gpt-oss-120b", "label": "DO GPT-oss-120b"},
    {"provider": "deepseek", "model": "deepseek-chat", "label": "DeepSeek Direct"},
    {"provider": "kimi", "model": "moonshot-v1-128k", "label": "Kimi Direct"},
    {"provider": "openai", "model": "gpt-4o", "label": "OpenAI Direct (Emergency)"},
    {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219", "label": "Anthropic Direct (Emergency)"},
]

# Default Tier 1 (Heavy Thinker / Architecture / RFC) Cascade Hierarchy
DEFAULT_THINKER_CASCADE_SPECS = [
    {"provider": "gemini", "model": "gemini-pro-latest", "label": "Gemini 3.1 Pro (Primary Credits)"},
    {"provider": "do", "model": "claude-opus-5", "label": "DO Claude Opus 5"},
    {"provider": "do", "model": "openai/gpt-5.6-sol", "label": "DO GPT-5.6 Sol"},
    {"provider": "anthropic", "model": "claude-3-opus-20240229", "label": "Anthropic Direct (Emergency)"},
    {"provider": "openai", "model": "o3-mini", "label": "OpenAI Direct (Emergency)"},
]

# Default Tier 3 (Worker Bee / Subagent / CUD / Grep) Cascade Hierarchy
DEFAULT_WORKER_CASCADE_SPECS = [
    {"provider": "gemini", "model": "gemini-flash-lite-latest", "label": "Gemini 3.5 Flash-Lite (Primary Credits)"},
    {"provider": "do", "model": "deepseek-v4-flash", "label": "DO DeepSeek V4 Flash"},
    {"provider": "do", "model": "openai/gpt-oss-120b", "label": "DO GPT-oss-120b"},
    {"provider": "do", "model": "openai/gpt-5-nano", "label": "DO GPT-5 Nano"},
    {"provider": "do", "model": "llama-4-maverick", "label": "DO Llama 4 Maverick"},
    {"provider": "deepseek", "model": "deepseek-chat", "label": "DeepSeek Direct"},
]


class FallbackTier:
    """Represents an instantiated provider and target model in the fallback chain."""

    def __init__(self, provider: BaseProvider, model: str, label: str, provider_name: str):
        self.provider = provider
        self.model = model
        self.label = label
        self.provider_name = provider_name

    def __repr__(self) -> str:
        return f"<FallbackTier label='{self.label}' provider='{self.provider_name}' model='{self.model}'>"


class ChainedFallbackProvider(BaseProvider):
    """Executes LLM turns across an ordered priority chain with automatic 503/429 failover."""

    def __init__(
        self,
        tiers: List[FallbackTier],
        default_model: Optional[str] = None,
        on_failover_callback: Optional[Callable[[FallbackTier, FallbackTier, BaseException], None]] = None,
        cooldown_sec: float = 60.0,
        **kwargs: Any,
    ) -> None:
        if not tiers:
            raise ValueError("ChainedFallbackProvider requires at least one active FallbackTier.")
        primary_model = default_model or tiers[0].model
        super().__init__(default_model=primary_model, **kwargs)
        self.tiers = tiers
        self.on_failover_callback = on_failover_callback
        self.cooldown_sec = cooldown_sec
        self._tier_cooldowns: Dict[int, float] = {}
        self._current_tier_index: int = 0

    def set_request_context(self, **context: Any) -> None:
        super().set_request_context(**context)
        for tier in self.tiers:
            setter = getattr(tier.provider, "set_request_context", None)
            if callable(setter):
                setter(**context)

    def reset_active_tier(self) -> None:
        """Resets tier index to primary (0) and clears all failure cooldowns."""
        self._current_tier_index = 0
        self._tier_cooldowns.clear()

    @property
    def active_tier(self) -> FallbackTier:
        if 0 <= self._current_tier_index < len(self.tiers):
            return self.tiers[self._current_tier_index]
        return self.tiers[0]

    @property
    def provider_name(self) -> str:
        return self.active_tier.provider_name

    @property
    def default_model(self) -> str:
        if self.tiers and self.active_tier:
            return self.active_tier.model
        return getattr(self, "_default_model", "gemini-flash-latest")

    @default_model.setter
    def default_model(self, value: Optional[str]) -> None:
        self._default_model = value

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[StreamEvent, Tuple[Message, Optional[UsageMetadata]], Message]]:
        """Attempts generation through the tier chain, catching provider errors and failing over."""
        collected_errors: List[str] = []

        now = time.time()
        start_index = 0
        # If no explicit model override was requested, skip tiers currently in failure cooldown
        if not model:
            while start_index < len(self.tiers) - 1 and self._tier_cooldowns.get(start_index, 0) > now:
                start_index += 1
            if start_index > 0:
                logger.info(
                    f"Tier 1..{start_index} in active failure cooldown ({self.cooldown_sec}s); "
                    f"starting turn at Tier {start_index + 1}: {self.tiers[start_index].label}"
                )

        for index in range(start_index, len(self.tiers)):
            tier = self.tiers[index]
            self._current_tier_index = index
            target_model = model if (index == 0 and model) else tier.model
            logger.info(f"Attempting turn via Tier {index + 1}/{len(self.tiers)}: {tier.label} ({target_model})")

            try:
                # Check if provider supports streaming
                if hasattr(tier.provider, "generate_stream"):
                    stream = tier.provider.generate_stream(
                        messages=messages,
                        tools=tools,
                        model=target_model,
                        **kwargs,
                    )
                    if inspect.isawaitable(stream):
                        stream = await stream

                    if hasattr(stream, "__aiter__"):
                        async for item in stream:
                            yield item
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!
                    elif hasattr(stream, "__iter__"):
                        for item in stream:
                            yield item
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!
                    else:
                        yield stream
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!

                else:
                    # Non-streaming provider execution
                    if inspect.iscoroutinefunction(tier.provider.generate):
                        res = await tier.provider.generate(
                            messages=messages,
                            tools=tools,
                            model=target_model,
                            **kwargs,
                        )
                    else:
                        res = tier.provider.generate(
                            messages=messages,
                            tools=tools,
                            model=target_model,
                            **kwargs,
                        )
                        if inspect.isawaitable(res):
                            res = await res

                    if isinstance(res, tuple):
                        final_msg, final_usage = res[0], res[1]
                    elif isinstance(res, Message):
                        final_msg, final_usage = res, getattr(res, "usage", None)
                    else:
                        final_msg, final_usage = None, None

                    if final_msg:
                        if not final_msg.provider:
                            final_msg.provider = tier.provider_name
                        if not final_msg.model:
                            final_msg.model = target_model
                        if final_msg.content:
                            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=final_msg.content)
                        if final_msg.tool_calls:
                            for tc in final_msg.tool_calls:
                                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
                        yield StreamEvent(
                            event_type=StreamEventType.TURN_COMPLETE,
                            message=final_msg,
                            usage=final_usage,
                        )
                    else:
                        yield res
                    self._tier_cooldowns.pop(index, None)
                    return  # Success!

            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                can_failover = is_provider_failover_candidate(exc)
                has_next_tier = index < (len(self.tiers) - 1)
                err_summary = f"Tier {index + 1} ({tier.label} / {target_model}) failed: {type(exc).__name__}: {str(exc)}"
                logger.warning(err_summary)
                collected_errors.append(err_summary)

                # Set cooldown for this failed tier so subsequent tool steps in this run don't retry it
                self._tier_cooldowns[index] = time.time() + self.cooldown_sec

                if can_failover and has_next_tier:
                    next_tier = self.tiers[index + 1]
                    self._current_tier_index = index + 1
                    logger.info(f"Cascading from {tier.label} to {next_tier.label} due to {type(exc).__name__}: {exc}")

                    if self.on_failover_callback:
                        try:
                            self.on_failover_callback(tier, next_tier, exc)
                        except Exception:
                            pass

                    # Clean error preview without massive raw dumps
                    err_clean = str(exc)
                    if len(err_clean) > 120:
                        err_clean = err_clean[:120] + "..."

                    # Emit visual notice chunk into stream
                    failover_msg = (
                        f"\n\033[93m⚠️ [Provider Failover]: {tier.label} returned {type(exc).__name__} ({err_clean}). "
                        f"Cascading to {next_tier.label} ({next_tier.model})...\033[0m\n"
                    )
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_CHUNK,
                        delta=failover_msg,
                        metadata={
                            "failover": True,
                            "failed_tier": tier.label,
                            "failed_model": target_model,
                            "next_tier": next_tier.label,
                            "next_model": next_tier.model,
                            "error": str(exc),
                        },
                    )
                    continue  # Move to next tier in cascade

                # If last tier exhausted
                raise exc

        raise RuntimeError(f"All {len(self.tiers)} fallback tiers exhausted.\n" + "\n".join(collected_errors))

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, Optional[UsageMetadata]]:
        """Non-streaming generation wrapper with failover."""
        final_msg = None
        final_usage = None

        async for item in self.generate_stream(messages=messages, tools=tools, model=model, **kwargs):
            if isinstance(item, tuple) and len(item) >= 2:
                final_msg, final_usage = item[0], item[1]
            elif isinstance(item, Message):
                final_msg = item
            elif isinstance(item, StreamEvent) and item.event_type == StreamEventType.TURN_COMPLETE and item.message:
                final_msg = item.message
                final_usage = item.usage

        if final_msg is not None:
            return final_msg, final_usage
        raise RuntimeError("ChainedFallbackProvider did not yield a valid Message response.")


def infer_cascade_tier(
    model_name: Optional[str] = None,
    profile_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Infers the appropriate fallback cascade tier (reasoner, thinker, worker)."""
    if profile_cfg and profile_cfg.get("model_tier"):
        raw_tier = str(profile_cfg["model_tier"]).lower()
        if "1" in raw_tier or "think" in raw_tier or "arch" in raw_tier:
            return "thinker"
        if "3" in raw_tier or "work" in raw_tier or "micro" in raw_tier:
            return "worker"
        return "reasoner"

    if model_name:
        m = model_name.lower()
        if any(w in m for w in ("lite", "nano", "mini", "flash-lite")):
            return "worker"
        if any(t in m for t in ("opus", "o1", "o3-mini", "o3")) or ("pro" in m and "flash" not in m):
            return "thinker"

    return "reasoner"


def build_fallback_chain(
    tier: str = "reasoner",
    primary_provider: Optional[str] = None,
    primary_model: Optional[str] = None,
    keystore: Optional[KeyStore] = None,
    custom_specs: Optional[List[Dict[str, Any]]] = None,
    enable_fallback: bool = True,
    **kwargs: Any,
) -> ChainedFallbackProvider:
    """Factory creating an active ChainedFallbackProvider rooted at your requested primary provider/model.

    If primary_provider is supplied, Tier 1 is always the requested primary model.
    Secondary and subsequent tiers form an automatic failover safety net based on available keys
    (DigitalOcean, Gemini, OpenAI, Anthropic) without requiring 'fallback' to be specified as provider.
    """
    from providers import resolve_provider_instance

    ks = keystore or KeyStore()
    t_norm = tier.strip().lower()

    # Check env override
    disable_env = os.environ.get("VALSTORM_DISABLE_FALLBACK", "").strip().lower() in ("1", "true", "yes")
    effective_enable_fallback = enable_fallback and not disable_env

    if custom_specs:
        specs = custom_specs
    elif t_norm in ("thinker", "architect", "tier1"):
        specs = DEFAULT_THINKER_CASCADE_SPECS
    elif t_norm in ("worker", "micro", "tier3"):
        specs = DEFAULT_WORKER_CASCADE_SPECS
    else:
        specs = DEFAULT_REASONER_CASCADE_SPECS

    active_tiers: List[FallbackTier] = []

    # 1. If primary_provider is explicitly requested (e.g. 'gemini', 'anthropic', 'openai', 'do')
    p_norm = (primary_provider or "").strip().lower()
    tier1_model = primary_model
    explicit_key = kwargs.pop("api_key", None)
    if p_norm and p_norm not in ("fallback", "cascade", "chained"):
        primary_key = explicit_key
        if not primary_key and p_norm == "valstorm":
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                primary_key, _ = resolve_valstorm_credentials()
            except Exception:
                pass
        if not primary_key:
            primary_key = ks.get_api_key(p_norm)
        if not primary_key and p_norm in ("do", "digitalocean"):
            primary_key = ks.get_api_key("digitalocean") or ks.get_api_key("do")

        try:
            primary_inst, resolved_model, resolved_p_name = resolve_provider_instance(
                provider_name=p_norm,
                api_key=primary_key,
                model=primary_model,
                **kwargs,
            )
            tier1_model = resolved_model
            active_tiers.append(
                FallbackTier(
                    provider=primary_inst,
                    model=resolved_model,
                    label=f"{p_norm.title()} Primary ({resolved_model})",
                    provider_name=resolved_p_name,
                )
            )
        except Exception as err:
            logger.warning(f"Could not initialize primary provider '{p_norm}': {err}")

    # 2. Append cascade backup tiers if fallback is enabled
    if effective_enable_fallback:
        for spec in specs:
            spec_p = spec["provider"].lower()
            spec_m = spec["model"].lower()

            # Skip if same provider family as primary (e.g. don't failover from Gemini to Gemini)
            if p_norm and spec_p == p_norm:
                continue
            if p_norm in ("do", "digitalocean") and spec_p in ("do", "digitalocean"):
                continue

            target_model = spec["model"]
            label = spec.get("label", f"{spec_p}/{target_model}")

            api_key = ks.get_api_key(spec_p)
            if not api_key and spec_p in ("do", "digitalocean"):
                api_key = ks.get_api_key("digitalocean") or ks.get_api_key("do")

            if not api_key:
                logger.debug(f"Skipping tier '{label}' - no API key configured for '{spec_p}'.")
                continue

            try:
                inst, resolved_model, resolved_p_name = resolve_provider_instance(
                    provider_name=spec_p,
                    api_key=api_key,
                    model=target_model,
                    **kwargs,
                )
                active_tiers.append(
                    FallbackTier(
                        provider=inst,
                        model=resolved_model,
                        label=label,
                        provider_name=resolved_p_name,
                    )
                )
            except Exception as err:
                logger.warning(f"Could not initialize provider for tier '{label}': {err}")

    # 3. If no tiers could be initialized (e.g. no keys or primary failed to construct)
    if not active_tiers:
        # Fallback to default gemini even without verified key so error message indicates missing key
        default_inst, d_model, d_p = resolve_provider_instance("gemini", api_key=None, model="gemini-flash-latest")
        active_tiers.append(
            FallbackTier(
                provider=default_inst,
                model=d_model,
                label="Gemini Default (Key Required)",
                provider_name=d_p,
            )
        )

    return ChainedFallbackProvider(tiers=active_tiers, default_model=active_tiers[0].model, **kwargs)
