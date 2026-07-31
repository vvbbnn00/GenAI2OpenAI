"""Backward-compatible import path for the chat service."""

import time as time

import requests as requests

from genai_proxy.api.openai.service import GenAIService as GenAIService
from genai_proxy.chat.service import *  # noqa: F403
from genai_proxy.chat.tool_loop import (
    _tool_start_tags_for_request as _tool_start_tags_for_request,
)
