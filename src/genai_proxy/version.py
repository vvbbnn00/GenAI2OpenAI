import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BUILD_METADATA_VERSION = 1
LOCAL_DEV_VERSION = "local-dev"
_BUILD_METADATA_PATH = Path(__file__).with_name("_build_version.json")
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True, slots=True)
class ProgramVersion:
    commit: str
    committed_at: str | None
    source: str
    dirty: bool = False

    @property
    def is_local_dev(self) -> bool:
        return self.commit == LOCAL_DEV_VERSION


def get_program_version(
    *,
    metadata_path: str | Path | None = None,
    source_root: str | Path | None = None,
) -> ProgramVersion:
    baked = _read_build_metadata(
        Path(metadata_path) if metadata_path is not None else _BUILD_METADATA_PATH
    )
    if baked is not None:
        return baked

    git_version = _read_git_version(
        Path(source_root) if source_root is not None else _SOURCE_ROOT
    )
    if git_version is not None:
        return git_version

    return ProgramVersion(LOCAL_DEV_VERSION, None, "fallback")


def format_program_version(version: ProgramVersion) -> str:
    if version.is_local_dev:
        return LOCAL_DEV_VERSION
    formatted = (
        f"commit={version.commit} "
        f"committed_at={version.committed_at} "
        f"source={version.source}"
    )
    if version.dirty:
        formatted += " dirty=true"
    return formatted


def write_build_metadata(
    output_path: str | Path,
    *,
    source_root: str | Path,
    environ: dict[str, str] | None = None,
) -> ProgramVersion:
    environment = os.environ if environ is None else environ
    dirty = _parse_build_dirty(environment.get("GENAI_BUILD_DIRTY"))
    version = None
    if dirty is not None:
        version = _version_from_values(
            environment.get("GENAI_BUILD_COMMIT"),
            environment.get("GENAI_BUILD_COMMIT_TIME"),
            source="build-args",
            dirty=dirty,
        )
    if version is None:
        version = _read_git_version(Path(source_root))
    if version is None:
        version = ProgramVersion(LOCAL_DEV_VERSION, None, "fallback")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BUILD_METADATA_VERSION,
        "commit": version.commit,
        "committed_at": version.committed_at,
        "dirty": version.dirty,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return version


def _read_build_metadata(path: Path) -> ProgramVersion | None:
    try:
        with path.open(encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
    except (FileNotFoundError, OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != BUILD_METADATA_VERSION:
        return None
    if payload.get("commit") == LOCAL_DEV_VERSION:
        return ProgramVersion(LOCAL_DEV_VERSION, None, "image")
    dirty = payload.get("dirty", False)
    if not isinstance(dirty, bool):
        return None
    return _version_from_values(
        payload.get("commit"),
        payload.get("committed_at"),
        source="image",
        dirty=dirty,
    )


def _read_git_version(source_root: Path) -> ProgramVersion | None:
    source_root = source_root.resolve()
    repository_root = _run_git(source_root, "rev-parse", "--show-toplevel")
    if repository_root is None:
        return None
    try:
        if Path(repository_root).resolve() != source_root:
            return None
    except OSError:
        return None

    details = _run_git(source_root, "log", "-1", "--format=%H%n%cI")
    if details is None:
        return None
    lines = details.splitlines()
    if len(lines) != 2:
        return None
    dirty = _run_git(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".dockerignore",
        "Dockerfile",
        "README.md",
        "docker-compose.yml",
        "main.py",
        "pyproject.toml",
        "scripts/docker-compose.sh",
        "src/genai_proxy",
        "uv.lock",
    )
    return _version_from_values(
        lines[0],
        lines[1],
        source="git",
        dirty=dirty is not None,
    )


def _run_git(source_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _version_from_values(
    commit,
    committed_at,
    *,
    source: str,
    dirty: bool = False,
) -> ProgramVersion | None:
    if not isinstance(commit, str):
        return None
    commit = commit.lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        return None
    if not isinstance(committed_at, str):
        return None
    try:
        parsed_time = datetime.fromisoformat(committed_at)
    except ValueError:
        return None
    if parsed_time.tzinfo is None:
        return None
    return ProgramVersion(commit, committed_at, source, dirty)


def _parse_build_dirty(value) -> bool | None:
    if value is None:
        return False
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized in {"", "0", "false", "no"}:
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    return None


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    version = write_build_metadata(
        args.output,
        source_root=args.source_root,
    )
    print(format_program_version(version))


if __name__ == "__main__":
    _main()
