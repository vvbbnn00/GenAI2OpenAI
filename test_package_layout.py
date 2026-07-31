import importlib
import importlib.metadata
import sys
from pathlib import Path

import genai_proxy
import genai_proxy.cli
import genai_proxy.runtime
import main as legacy_main


def test_distribution_and_import_package_names_are_preserved():
    package_path = Path(genai_proxy.__file__).resolve()
    assert package_path.parent.name == "genai_proxy"
    assert package_path.parent.parent.name == "src"
    assert importlib.metadata.distribution("genai").metadata["Name"] == "genai"


def test_root_main_is_a_thin_cli_compatibility_entrypoint():
    assert legacy_main.main is genai_proxy.cli.main
    assert not hasattr(legacy_main, "app")


def test_wsgi_module_builds_its_app_through_the_runtime_factory(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(genai_proxy.runtime, "create_app_from_env", lambda: sentinel)
    sys.modules.pop("genai_proxy.wsgi", None)

    wsgi = importlib.import_module("genai_proxy.wsgi")
    try:
        assert wsgi.app is sentinel
    finally:
        sys.modules.pop("genai_proxy.wsgi", None)
