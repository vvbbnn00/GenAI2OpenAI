import ast
from pathlib import Path

import genai_proxy.compat.claude as legacy_anthropic
import genai_proxy.compat.openai as legacy_openai
import genai_proxy.compat.responses as legacy_responses
import genai_proxy.routes.claude as legacy_anthropic_routes
import genai_proxy.routes.openai as legacy_openai_routes
from genai_proxy.api.anthropic import compat as anthropic
from genai_proxy.api.anthropic.routes import bp as anthropic_bp
from genai_proxy.api.openai import responses
from genai_proxy.api.openai.errors import make_error_chunk, openai_error
from genai_proxy.api.openai.routes import bp as openai_bp
from genai_proxy.api.openai.service import GenAIService, OpenAIProtocolMixin
from genai_proxy.chat.service import ChatService
from genai_proxy.chat.tool_protocol import extract_tool_calls


def test_legacy_api_imports_forward_to_the_canonical_modules():
    assert legacy_anthropic.convert_claude_to_openai is (
        anthropic.convert_claude_to_openai
    )
    assert legacy_responses.convert_responses_to_openai_request is (
        responses.convert_responses_to_openai_request
    )
    assert legacy_openai.extract_tool_calls is extract_tool_calls
    assert legacy_openai.make_error_chunk is make_error_chunk
    assert legacy_openai.openai_error is openai_error
    assert legacy_anthropic_routes.bp is anthropic_bp
    assert legacy_openai_routes.bp is openai_bp


def test_openai_service_composes_protocol_and_chat_layers():
    assert issubclass(GenAIService, ChatService)
    assert GenAIService.build_openai_completion is (
        OpenAIProtocolMixin.build_openai_completion
    )
    assert not hasattr(ChatService, "build_openai_completion")


def test_chat_layer_does_not_import_api_or_compat_modules():
    chat_dir = Path(__file__).parent / "src" / "genai_proxy" / "chat"
    forbidden_prefixes = ("genai_proxy.api", "genai_proxy.compat")
    assert chat_dir.is_dir()

    for path in chat_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        assert not any(
            module.startswith(forbidden_prefixes) for module in imported_modules
        ), path
