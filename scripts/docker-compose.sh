#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd)
cd "${project_dir}"

GENAI_BUILD_COMMIT=${GENAI_BUILD_COMMIT:-}
GENAI_BUILD_COMMIT_TIME=${GENAI_BUILD_COMMIT_TIME:-}
GENAI_BUILD_DIRTY=${GENAI_BUILD_DIRTY:-0}

git_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "${git_root}" ] && [ "$(CDPATH= cd -- "${git_root}" && pwd)" = "${project_dir}" ]; then
    GENAI_BUILD_COMMIT=$(git log -1 --format=%H)
    GENAI_BUILD_COMMIT_TIME=$(git log -1 --format=%cI)
    if [ -n "$(git status --porcelain --untracked-files=normal -- \
        .dockerignore Dockerfile README.md docker-compose.yml main.py \
        pyproject.toml scripts/docker-compose.sh src/genai_proxy uv.lock)" ]; then
        GENAI_BUILD_DIRTY=1
    fi
fi

export GENAI_BUILD_COMMIT
export GENAI_BUILD_COMMIT_TIME
export GENAI_BUILD_DIRTY

exec docker compose "$@"
