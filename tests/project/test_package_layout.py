import importlib
import importlib.metadata
import importlib.util
import sys
import tomllib
from pathlib import Path

import genai_proxy
import genai_proxy.cli
import genai_proxy.runtime


def _load_root_main():
    path = Path(__file__).resolve().parents[2] / "main.py"
    spec = importlib.util.spec_from_file_location("genai2openai_root_main", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distribution_and_import_package_names_are_preserved():
    package_path = Path(genai_proxy.__file__).resolve()
    assert package_path.parent.name == "genai_proxy"
    assert package_path.parent.parent.name == "src"
    assert importlib.metadata.distribution("genai").metadata["Name"] == "genai"


def test_root_main_is_a_thin_cli_compatibility_entrypoint():
    legacy_main = _load_root_main()
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


def test_wiki_is_shipped_in_sdist_but_excluded_from_runtime_artifacts():
    project_root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert "docs" in project["tool"]["hatch"]["build"]["targets"]["sdist"][
        "include"
    ]
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/genai_proxy"
    ]
    assert "COPY docs" not in (project_root / "Dockerfile").read_text()
    assert "docs" in (project_root / ".dockerignore").read_text().splitlines()
