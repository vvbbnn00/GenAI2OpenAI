"""Backward-compatible import path for the chat service."""

import time

import requests

from genai_proxy.chat.service import *  # noqa: F403
from genai_proxy.chat.service import _tool_start_tags_for_request
