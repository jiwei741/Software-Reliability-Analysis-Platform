# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import requests

DEFAULT_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


class DeepSeekError(RuntimeError):
    pass


def _build_prompt(headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> str:
    sample_lines = []
    sample_lines.append(",".join(headers))
    for row in rows[:5]:
        sample_lines.append(",".join(str(row.get(header, "")) for header in headers))
    sample = "\n" + "\n".join(sample_lines)
    instructions = (
        "请根据提供的表头和样例数据，推断哪一列对应 'module'(模块)、'failures'(失效次数)、"
        "'mtbf'(平均失效间隔)、'runtime'(运行时长)。"
        "如果缺失某一列，请返回空字符串。务必输出 JSON，格式如下：\n"
        "{\"module\": \"列名\", \"failures\": \"列名\", \"mtbf\": \"列名\", \"runtime\": \"列名\"}."
    )
    return f"{instructions}\n样例：{sample}"


def infer_mapping_from_sample(
    headers: Sequence[str],
    rows: Sequence[Dict[str, object]],
    logger=None,
) -> Dict[str, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY 未配置")
    if not headers:
        raise DeepSeekError("无法获取表头")

    prompt = _build_prompt(headers, rows)
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a data expert that responds strictly with JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(
            DEFAULT_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover
        raise DeepSeekError(f"调用 DeepSeek 失败: {exc}") from exc

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, ValueError, IndexError) as exc:
        raise DeepSeekError("无法解析 DeepSeek 响应") from exc

    mapping = None
    try:
        mapping = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                mapping = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                mapping = None
    if not isinstance(mapping, dict):
        raise DeepSeekError("DeepSeek 未返回合法 JSON")

    result = {}
    for key in ("module", "failures", "mtbf", "runtime"):
        value = mapping.get(key)
        if isinstance(value, str):
            result[key] = value.strip()
        else:
            result[key] = ""
    if logger:
        logger.info("DeepSeek 映射结果: %s", result)
    return result
