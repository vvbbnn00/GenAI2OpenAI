"""Canonical request preparation before token counting or transport."""

import json

from genai_proxy.chat.tool_choice import tool_choice_is_none as _tool_choice_is_none
from genai_proxy.chat.tool_protocol import inject_tool_prompt
from genai_proxy.chat.types import PreparedChatRequest, ResolvedModelContext
from genai_proxy.errors import ProxyError
from genai_proxy.messages import adapter_supports_vision, normalize_message_contents
from genai_proxy.models import (
    DEEPSEEK_V4_ADAPTERS,
    GLM_5_2_ADAPTER,
    KIMI_K3_ADAPTER,
    KIMI_TOOL_TRANSPORT_ERROR,
    inject_deepseek_reasoning_prompt,
    inject_glm_reasoning_prompt,
    inject_kimi_tool_prompt,
    select_tool_adapter,
)
from genai_proxy.reasoning import (
    deepseek_thinking_enabled,
    normalize_reasoning_for_adapter,
    parse_reasoning_config,
)
from genai_proxy.token_usage import (
    count_openai_request_tokens,
    kimi_image_sizes_for_messages,
    tokenizer_family_for_model,
)

KIMI_EMPTY_CURRENT_INPUT = "\u200b"


class ChatPreparationMixin:
    def resolve_model_context(self, requested_model: str) -> ResolvedModelContext:
        if not isinstance(requested_model, str):
            raise ProxyError("'model' must be a string")
        resolve_model_record = getattr(
            self._model_manager,
            "resolve_model_record",
            None,
        )
        if callable(resolve_model_record):
            model, model_record = resolve_model_record(requested_model)
        else:
            model = self._model_manager.resolve_model(requested_model)
            model_record = self._model_manager.get_model_record(model)
        tool_adapter = select_tool_adapter(model, model_record)
        root_ai_type = (model_record or {}).get("rootAiType")
        if not root_ai_type:
            root_ai_type = self._model_manager.root_ai_type_for(model)
        return ResolvedModelContext(
            requested_model=requested_model,
            model=model,
            model_record=model_record,
            tool_adapter=tool_adapter,
            tokenizer_family=tokenizer_family_for_model(
                model,
                model_record,
                tool_adapter,
            ),
            supports_vision=adapter_supports_vision(tool_adapter),
            supports_thinking_toggle=tool_adapter in DEEPSEEK_V4_ADAPTERS,
            transport=(
                "kimi_web" if tool_adapter == KIMI_K3_ADAPTER else "genai_chat"
            ),
            root_ai_type=str(root_ai_type),
            root_model_name=(model_record or {}).get("rootModelName"),
        )

    def _prepare_chat_request(
        self,
        req_data,
        *,
        count_usage: bool = True,
        model_context: ResolvedModelContext | None = None,
    ) -> PreparedChatRequest:
        if not req_data or "messages" not in req_data:
            raise ProxyError("Missing 'messages' field in request body")

        messages = req_data.get("messages", [])
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) for message in messages
        ):
            raise ProxyError("'messages' must be a list of objects")
        _validate_openai_message_shapes(messages)

        requested_model = req_data.get("model", "GPT-4.1")
        if not isinstance(requested_model, str):
            raise ProxyError("'model' must be a string")
        if model_context is None:
            model_context = self.resolve_model_context(requested_model)
        elif requested_model.casefold() not in {
            model_context.requested_model.casefold(),
            model_context.model.casefold(),
        }:
            raise RuntimeError("Resolved model context does not match the request")
        model = model_context.model
        max_tokens = req_data.get("max_tokens", 30000)
        tools = req_data.get("tools") or []
        if not isinstance(tools, list) or any(
            not isinstance(tool, dict) for tool in tools
        ):
            raise ProxyError("'tools' must be a list of objects")
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                raise ProxyError("Function tools must contain a 'function' object")
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise ProxyError("Function tools must contain a non-empty string name")
            if function.get("parameters") is not None and not isinstance(
                function["parameters"], dict
            ):
                raise ProxyError("Function tool parameters must be an object")
        tool_choice = req_data.get("tool_choice")
        model_record = model_context.model_record
        tool_adapter = model_context.tool_adapter
        messages = normalize_message_contents(messages, adapter=tool_adapter)
        messages = _normalize_messages_for_model_template(
            messages,
            model,
            model_record=model_record,
            tool_adapter=tool_adapter,
        )
        reasoning_config = parse_reasoning_config(req_data)
        reasoning_config = normalize_reasoning_for_adapter(
            tool_adapter, reasoning_config
        )
        thinking = deepseek_thinking_enabled(tool_adapter, reasoning_config)

        requested_tools = bool(tools)
        has_tools = requested_tools and not _tool_choice_is_none(tool_choice)
        if requested_tools:
            messages = inject_tool_prompt(
                messages,
                tools,
                tool_choice,
                model=model,
                adapter=tool_adapter,
                reasoning_config=reasoning_config,
            )
        elif tool_adapter == KIMI_K3_ADAPTER and any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in messages
        ):
            messages = inject_kimi_tool_prompt(messages, [], tool_choice="none")
        elif tool_adapter == GLM_5_2_ADAPTER:
            messages = inject_glm_reasoning_prompt(messages, reasoning_config)
        elif tool_adapter in DEEPSEEK_V4_ADAPTERS:
            messages = inject_deepseek_reasoning_prompt(
                messages,
                reasoning_config,
                adapter=tool_adapter,
            )

        if tool_adapter == KIMI_K3_ADAPTER:
            messages = normalize_kimi_messages_for_transport(messages)

        if not self._extract_last_user_message(messages):
            raise ProxyError("No user message found in 'messages'")

        # The transport carries two-level reasoning as injected message text.
        # Passing it to the template again would count a different prompt.
        token_reasoning_config = (
            None
            if tool_adapter == GLM_5_2_ADAPTER or tool_adapter in DEEPSEEK_V4_ADAPTERS
            else reasoning_config
        )
        include_usage = bool(
            isinstance(req_data.get("stream_options"), dict)
            and req_data["stream_options"].get("include_usage")
        )
        prompt_tokens = None
        image_sizes = None
        family = model_context.tokenizer_family
        if family == "kimi_k3":
            image_sizes = kimi_image_sizes_for_messages(messages)
        if count_usage:
            prompt_tokens = count_openai_request_tokens(
                messages,
                model,
                model_record=model_record,
                tool_adapter=tool_adapter,
                reasoning_config=token_reasoning_config,
                thinking=thinking,
                image_sizes=image_sizes,
            )

        return PreparedChatRequest(
            messages=messages,
            model=model,
            root_model_name=model_context.root_model_name,
            root_ai_type=model_context.root_ai_type,
            max_tokens=max_tokens,
            has_tools=has_tools,
            tools=tools if has_tools else [],
            tool_choice=tool_choice if has_tools else None,
            tool_adapter=tool_adapter,
            model_record=model_record,
            include_usage=include_usage,
            prompt_tokens=prompt_tokens,
            token_reasoning_config=token_reasoning_config,
            thinking=thinking,
            image_sizes=image_sizes,
        )

    def _extract_last_user_message(self, messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False)
        return ""


def normalize_kimi_messages_for_transport(messages: list[dict]) -> list[dict]:
    if any(
        message.get("role") not in {"system", "user", "assistant", "tool"}
        for message in messages
    ):
        raise ProxyError("Kimi K3 received an unsupported message role")
    for message in messages:
        _validate_kimi_message_content(message)
        if message.get("tools"):
            raise ProxyError(
                KIMI_TOOL_TRANSPORT_ERROR,
                error_type="invalid_request_error",
                code="unsupported_tool_transport",
                status=400,
            )
        if message.get("role") == "tool" or message.get("tool_calls"):
            raise ProxyError(
                "Kimi K3 tool history cannot be forwarded through the "
                "ShanghaiTech GenAI transport without changing Moonshot's "
                "official XTML encoding",
                error_type="invalid_request_error",
                code="unsupported_tool_transport",
                status=400,
            )
    messages = [
        {**message, "content": ""} if message.get("content") is None else message
        for message in messages
    ]
    if not messages or not any(message.get("role") == "user" for message in messages):
        return messages
    if messages[-1].get("role") != "user":
        return [
            *messages,
            {"role": "user", "content": KIMI_EMPTY_CURRENT_INPUT},
        ]

    content = messages[-1].get("content")
    if isinstance(content, str):
        if content:
            return [*messages[:-1], _kimi_current_user_message(content)]
        return [
            *messages[:-1],
            _kimi_current_user_message(KIMI_EMPTY_CURRENT_INPUT),
        ]
    if not isinstance(content, list):
        raise ProxyError(
            "Kimi K3 message content must be a string or a list of content parts"
        )

    text_parts = []
    image_parts = []
    for part in content:
        if part.get("type") in {"image", "image_url"}:
            image_parts.append(part)
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise ProxyError(
                "Kimi K3 message content supports only text and image parts"
            )
        text_parts.append(text)

    text = "".join(text_parts)
    if not image_parts:
        normalized_content = text or KIMI_EMPTY_CURRENT_INPUT
    else:
        normalized_content = [
            {
                "type": "text",
                "text": text or KIMI_EMPTY_CURRENT_INPUT,
            },
            *image_parts,
        ]

    normalized_message = _kimi_current_user_message(normalized_content)
    return [*messages[:-1], normalized_message]


def _validate_kimi_message_content(message: dict) -> None:
    role = message.get("role")
    content = message.get("content")
    if role == "tool" or content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ProxyError(
            "Kimi K3 message content must be a string or a list of content parts"
        )
    for part in content:
        part_type = part.get("type")
        if part_type in {"image", "image_url"}:
            if role != "user":
                raise ProxyError("Kimi K3 accepts image content only in user messages")
            continue
        if part_type != "text" or not isinstance(part.get("text"), str):
            raise ProxyError(
                "Kimi K3 message content supports only text and image parts"
            )


def _kimi_current_user_message(content) -> dict:
    return {"role": "user", "content": content}


def genai_transport_input(
    messages: list[dict],
    tool_adapter: str,
    image_sizes: tuple[tuple[int, int], ...] | None,
):
    if tool_adapter != KIMI_K3_ADAPTER:
        return messages, "", {}

    current = messages[-1]
    if current.get("role") != "user":
        raise RuntimeError("Kimi K3 transport requires a final user message")

    content = current.get("content")
    image_urls = []
    if isinstance(content, str):
        chat_info = content
    else:
        text_parts = []
        for part in content or []:
            if part.get("type") in {"image", "image_url"}:
                image_urls.append(_kimi_image_url(part))
            else:
                text_parts.append(str(part.get("text") or ""))
        chat_info = "".join(text_parts) or KIMI_EMPTY_CURRENT_INPUT

    image_fields = {}
    if image_urls:
        current_sizes = (image_sizes or ())[-len(image_urls) :]
        if len(current_sizes) != len(image_urls):
            raise RuntimeError("Kimi K3 image dimensions were not prepared")
        width, height = current_sizes[0]
        image_fields = {
            "imageUrl": image_urls[0],
            "imageUrls": image_urls,
            "width": width,
            "height": height,
        }

    return messages[:-1], chat_info, image_fields


def _kimi_image_url(part: dict) -> str:
    source = part.get(part.get("type"))
    if source is None and part.get("type") == "image":
        source = part.get("url")
    if isinstance(source, dict):
        source = source.get("url", source.get("data"))
    if not isinstance(source, str) or not source:
        raise ProxyError("Kimi K3 image content is missing its URL")
    return source


def _validate_openai_message_shapes(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            not isinstance(part, dict) for part in content
        ):
            raise ProxyError("Message content arrays must contain objects")

        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list) or any(
            not isinstance(tool_call, dict) for tool_call in tool_calls
        ):
            raise ProxyError("'tool_calls' must be a list of objects")
        for tool_call in tool_calls:
            if not isinstance(tool_call.get("function"), dict):
                raise ProxyError("Each tool call must contain a 'function' object")


def _normalize_messages_for_model_template(
    messages: list[dict],
    model: str,
    *,
    model_record: dict | None,
    tool_adapter: str,
) -> list[dict]:
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family not in {
        "glm_5_1",
        "glm_5_2",
        "qwen_3_5",
        "minimax_m2_7",
        "kimi_k3",
    }:
        return messages

    normalized = [
        {**message, "role": "system"}
        if message.get("role") == "developer"
        else message
        for message in messages
    ]
    if family not in {"qwen_3_5", "minimax_m2_7"}:
        return normalized

    system_messages = [
        message for message in normalized if message.get("role") == "system"
    ]
    if not system_messages:
        return normalized
    non_system_messages = [
        message for message in normalized if message.get("role") != "system"
    ]
    return [
        {
            "role": "system",
            "content": _merge_system_contents(system_messages),
        },
        *non_system_messages,
    ]


def _merge_system_contents(messages: list[dict]):
    contents = [message.get("content", "") for message in messages]
    if all(isinstance(content, str) for content in contents):
        return "\n\n".join(contents)

    parts = []
    for index, content in enumerate(contents):
        if index:
            parts.append({"type": "text", "text": "\n\n"})
        if isinstance(content, list):
            parts.extend(content)
        else:
            parts.append({"type": "text", "text": str(content or "")})
    return parts


__all__ = [
    "ChatPreparationMixin",
    "KIMI_EMPTY_CURRENT_INPUT",
    "genai_transport_input",
    "normalize_kimi_messages_for_transport",
]
