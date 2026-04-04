FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple

WORKDIR /app

RUN apt-get update \
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
    idna==3.11 \
    itsdangerous==2.2.0 \
    jinja2==3.1.6 \
    markupsafe==3.0.3 \
    pycparser==3.0 \
    requests==2.32.5 \
    urllib3==2.5.0 \
    werkzeug==3.1.3 \
    "shanghaitech-ids-passkey @ git+https://github.com/vvbbnn00/shanghaitech-ids-passkey.git@7c4df62716ceb3d94452d22f3d07f19ff1b8db8b"

COPY pyproject.toml uv.lock README.md ./
COPY genai_proxy ./genai_proxy
COPY main.py ./

EXPOSE 5000
