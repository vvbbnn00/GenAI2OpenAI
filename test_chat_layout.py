import requests
import time

import genai_proxy.services.genai as legacy_chat
from genai_proxy.chat.preparation import ChatPreparationMixin
from genai_proxy.chat.service import GenAIService
from genai_proxy.chat.streaming import ChatStreamingMixin
from genai_proxy.chat.tool_loop import ToolLoopMixin
from genai_proxy.chat.usage import ChatUsageMixin


def test_legacy_chat_service_import_preserves_identity_and_patch_modules():
    assert legacy_chat.GenAIService is GenAIService
    assert legacy_chat.requests is requests
    assert legacy_chat.time is time


def test_chat_service_uses_the_split_implementations():
    assert GenAIService._prepare_chat_request is (
        ChatPreparationMixin._prepare_chat_request
    )
    assert GenAIService._usage is ChatUsageMixin._usage
    assert GenAIService._stream_genai_response_raw is (
        ChatStreamingMixin._stream_genai_response_raw
    )
    assert GenAIService._stream_genai_response_with_tools is (
        ToolLoopMixin._stream_genai_response_with_tools
    )
