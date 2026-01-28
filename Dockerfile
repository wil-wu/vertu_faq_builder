FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 设置时区为中国时区
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai

COPY . /vertu/src

ENV UV_NO_DEV=1

WORKDIR /vertu/src
RUN uv sync --locked

CMD ["uv", "run", "main.py"]
