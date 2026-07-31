FROM python:3.11-slim

ARG DEBIAN_MIRROR=http://mirrors.ustc.edu.cn/debian
ARG DEBIAN_SECURITY_MIRROR=http://mirrors.ustc.edu.cn/debian-security

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple

WORKDIR /app

RUN set -eux; \
    for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
        [ -f "$source_file" ] || continue; \
        sed -i \
            -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
            -e "s|https://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
            -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
            -e "s|https://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
            -e "s|http://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
            -e "s|https://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
            "$source_file"; \
    done \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    blinker==1.9.0 \
    certifi==2025.11.12 \
    cffi==2.0.0 \
    charset-normalizer==3.4.4 \
    click==8.3.1 \
    cryptography==46.0.6 \
    flask==3.1.2 \
    flask-cors==6.0.1 \
    gunicorn==23.0.0 \
    idna==3.11 \
    itsdangerous==2.2.0 \
    jinja2==3.1.6 \
    markupsafe==3.0.3 \
    pillow==12.3.0 \
    pycparser==3.0 \
    regex==2026.7.19 \
    requests==2.32.5 \
    tiktoken==0.12.0 \
    tokenizers==0.22.2 \
    urllib3==2.5.0 \
    werkzeug==3.1.3 \
    "shanghaitech-ids-passkey @ git+https://github.com/vvbbnn00/shanghaitech-ids-passkey.git@7c4df62716ceb3d94452d22f3d07f19ff1b8db8b"

COPY pyproject.toml uv.lock README.md ./
COPY genai_proxy ./genai_proxy
COPY main.py ./

ARG GENAI_BUILD_COMMIT=""
ARG GENAI_BUILD_COMMIT_TIME=""
ARG GENAI_BUILD_DIRTY="0"
RUN GENAI_BUILD_COMMIT="${GENAI_BUILD_COMMIT}" \
    GENAI_BUILD_COMMIT_TIME="${GENAI_BUILD_COMMIT_TIME}" \
    GENAI_BUILD_DIRTY="${GENAI_BUILD_DIRTY}" \
    python -m genai_proxy.version \
        --output ./genai_proxy/_build_version.json \
        --source-root /app

EXPOSE 5000
