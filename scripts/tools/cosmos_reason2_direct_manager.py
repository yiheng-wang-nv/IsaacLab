"""Cosmos Reason2 direct-manager client for trocar multi-stage inference."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"continue", "complete", "retry"}


@dataclass(frozen=True)
class CosmosDecision:
    is_wrong: bool
    decision: str
    task_index: int | None
    confidence: float
    reason: str
    raw_text: str

    @classmethod
    def from_text(cls, text: str) -> "CosmosDecision":
        payload = _extract_json_object(text)
        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid Cosmos decision {decision!r}; expected one of {sorted(VALID_DECISIONS)}.")

        is_wrong = _optional_bool(payload.get("is_wrong"), default=decision == "retry")
        task_index = _optional_int(payload.get("task_index"))
        confidence = _clamp_float(payload.get("confidence", 0.0), 0.0, 1.0)
        reason = str(payload.get("reason", "")).strip()
        return cls(
            is_wrong=is_wrong,
            decision=decision,
            task_index=task_index,
            confidence=confidence,
            reason=reason,
            raw_text=text,
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "is_wrong": self.is_wrong,
            "decision": self.decision,
            "task_index": self.task_index,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_text": self.raw_text,
        }


class CosmosReason2DirectManager:
    """OpenAI-compatible HTTP client for a vLLM-hosted Cosmos Reason2 model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        prompt_path: str | Path,
        timeout_s: float,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        prompt_config = _load_prompt_config(Path(prompt_path))
        self.system_prompt = prompt_config["system_prompt"]
        self.user_prompt_template = prompt_config["user_prompt"]

    def save_camera_snapshots(
        self,
        *,
        camera_frames: dict[str, Any],
        output_dir: str | Path,
        decision_index: int,
    ) -> dict[str, str]:
        from PIL import Image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, str] = {}
        for camera_name, frame in camera_frames.items():
            path = output_path / f"decision_{decision_index:06d}_{camera_name}.png"
            Image.fromarray(frame).save(path)
            saved_paths[camera_name] = str(path)
        return saved_paths

    def decide(
        self,
        *,
        context: dict[str, Any],
        image_paths_by_frame: list[dict[str, str]],
    ) -> CosmosDecision:
        messages = self._build_messages(context=context, image_paths_by_frame=image_paths_by_frame)
        response = self._post_chat_completion(messages)
        text = _response_text(response)
        return CosmosDecision.from_text(text)

    def _build_messages(
        self,
        *,
        context: dict[str, Any],
        image_paths_by_frame: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        ordered_images: list[tuple[str, str, str]] = []
        for frame_idx, paths_by_camera in enumerate(image_paths_by_frame):
            for camera_name, image_path in sorted(paths_by_camera.items()):
                ordered_images.append((f"frame_{frame_idx}", camera_name, Path(image_path).resolve().as_uri()))

        image_order = [
            {"frame": frame_label, "camera": camera_name, "image_index": idx}
            for idx, (frame_label, camera_name, _) in enumerate(ordered_images)
        ]
        prompt_context = dict(context)
        prompt_context["image_order"] = image_order
        user_prompt = self.user_prompt_template.format(
            state_json=json.dumps(prompt_context, indent=2, sort_keys=True)
        )

        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_url}}
            for _, _, image_url in ordered_images
        ]
        user_content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _post_chat_completion(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cosmos HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cosmos request failed: {exc}") from exc


def _load_prompt_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Cosmos prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            system_prompt = str(loaded.get("system_prompt", "")).strip()
            user_prompt = str(loaded.get("user_prompt", "")).strip()
            if system_prompt and user_prompt:
                return {"system_prompt": system_prompt, "user_prompt": user_prompt}
    except Exception:
        pass

    parsed = _parse_simple_yaml_blocks(text)
    if parsed.get("system_prompt") and parsed.get("user_prompt"):
        return parsed
    raise ValueError(f"Cosmos prompt file must define system_prompt and user_prompt: {path}")


def _parse_simple_yaml_blocks(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.match(r"^(system_prompt|user_prompt):\s*\|\s*$", line)
        if match is None:
            idx += 1
            continue
        key = match.group(1)
        idx += 1
        block: list[str] = []
        while idx < len(lines):
            block_line = lines[idx]
            if block_line and not block_line.startswith((" ", "\t")):
                break
            block.append(block_line[2:] if block_line.startswith("  ") else block_line.lstrip())
            idx += 1
        result[key] = "\n".join(block).strip()
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    json_text = _first_balanced_json_object(stripped)
    if json_text is None:
        raise ValueError(f"Cosmos response did not contain a JSON object: {text!r}")
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Cosmos response JSON must be an object: {text!r}")
    return payload


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not choices:
        raise ValueError(f"Cosmos response has no choices: {response}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    raise ValueError(f"Cosmos response message has no text content: {response}")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "wrong"}
    return bool(value)


def _clamp_float(value: Any, min_value: float, max_value: float) -> float:
    number = float(value)
    return max(min_value, min(max_value, number))
