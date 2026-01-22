FROM ghcr.io/astral-sh/uv:python3.12-alpine

# 设置时区为中国时区
RUN apk add --no-cache tzdata && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

ENV TZ=Asia/Shanghai

COPY . /vertu/src

ENV UV_NO_DEV=1

WORKDIR /vertu/src
RUN uv sync --locked

CMD ["uv", "run", "main.py"]
