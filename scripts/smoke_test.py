# -*- coding: utf-8 -*-
"""FinRAG API 冒烟测试：需先启动 uvicorn app:app。"""
import asyncio
import json
import sys

import requests
import websockets

BASE_URL = "http://127.0.0.1:8080"


def check_http(path, expected_errno=0):
    response = requests.get(BASE_URL + path, timeout=30)
    response.raise_for_status()
    payload = response.json()
    assert payload["errno"] == expected_errno, payload
    print(f"[OK] GET {path} -> {payload['data']}")


async def check_ws():
    async with websockets.connect(f"ws://127.0.0.1:8080/api/stream", max_size=10 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"query": "上市公司股东享有哪些权利？", "source_filter": "股票"}, ensure_ascii=False))
        tokens = []
        ended = False
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            if message["type"] == "token":
                tokens.append(message["token"])
            elif message["type"] == "end":
                ended = True
                break
            elif message["type"] == "error":
                raise RuntimeError(message)
        assert ended and "".join(tokens)
        print(f"[OK] WebSocket /api/stream -> {len(''.join(tokens))} chars")


def main():
    check_http("/health")
    check_http("/live")
    check_http("/ready")
    check_http("/api/sources")
    response = requests.post(BASE_URL + "/api/query", json={"query": "什么是基金定投", "source_filter": ""}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    assert payload["errno"] == 0, payload
    assert payload["data"]["answer"]
    print(f"[OK] POST /api/query -> {payload['data']['answer']}")
    asyncio.run(check_ws())


if __name__ == "__main__":
    main()
