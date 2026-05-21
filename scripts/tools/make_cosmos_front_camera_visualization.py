#!/usr/bin/env python3
"""Create front-camera + Cosmos decision-panel videos.

The left half shows the recorded front camera. The right half is a black
decision panel with timed subtitles from ``cosmos_decisions.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


TASK_NAMES = {
    0: "Task 1: left pickup",
    1: "Task 2: right pickup",
    2: "Task 3: align",
    3: "Task 4: install",
    4: "Task 5: place",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(
            "/home/nvidia/workspace/yiheng/IsaacLab/"
            "multistage_direct_videos_10e_cosmos_reason2_every60_rand_0_10"
        ),
        help="Cosmos run output directory containing episode folders and cosmos_decisions.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for rendered visualization videos.")
    parser.add_argument("--episode", type=int, action="append", default=None, help="Episode index to render. Repeatable.")
    parser.add_argument("--fps", type=float, default=30.0, help="Recorded video FPS.")
    parser.add_argument("--subtitle-duration", type=float, default=4.0, help="Seconds to display each decision subtitle.")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=480)
    return parser.parse_args()


def _read_decisions(path: Path) -> dict[int, list[dict]]:
    decisions: dict[int, list[dict]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing Cosmos decision log: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            decisions.setdefault(int(item["episode_idx"]), []).append(item)
    for items in decisions.values():
        items.sort(key=lambda item: int(item.get("decision_index", 0)))
    return decisions


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    value = result.stdout.strip()
    return float(value) if value and value != "N/A" else 0.0


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds >= 100:
        secs += 1
        centiseconds -= 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def _wrap(text: str, width: int = 42) -> str:
    return "\\N".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def _decision_text(item: dict) -> str:
    context = item.get("context", {})
    cosmos = item.get("cosmos", {})
    task_idx = int(item.get("task_idx", context.get("current_task_index", -1)))
    progress = context.get("complete_probability")
    threshold = context.get("threshold_hint")
    elapsed = context.get("task_elapsed_steps")
    timeout = context.get("timeout_steps")

    if "error" in cosmos:
        decision = "ERROR"
        reason = str(cosmos["error"])
        confidence = "n/a"
    else:
        decision = str(cosmos.get("decision", "unknown")).upper()
        reason = str(cosmos.get("reason", ""))
        confidence = cosmos.get("confidence", "n/a")

    progress_text = "n/a" if progress is None else f"{float(progress):.4f}"
    threshold_text = "n/a" if threshold is None else f"{float(threshold):.4f}"
    confidence_text = confidence if isinstance(confidence, str) else f"{float(confidence):.2f}"
    return "\n".join(
        [
            "ASK COSMOS @ timeout boundary",
            TASK_NAMES.get(task_idx, f"Task {task_idx + 1}"),
            f"step: {elapsed}/{timeout}",
            f"progress: {progress_text}   threshold: {threshold_text}",
            f"RESPONSE: {decision}   conf: {confidence_text}",
            f"reason: {_wrap(reason)}",
        ]
    )


def _write_ass(path: Path, episode_decisions: list[dict], *, fps: float, subtitle_duration: float, video_duration: float) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 480

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Panel,DejaVu Sans,24,&H00FFFFFF,&H00FFFFFF,&H00000000,&HAA000000,0,0,0,0,100,100,0,0,1,1,0,7,670,20,28,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for item in episode_decisions:
        step = float(item.get("total_env_steps", 0))
        start = step / fps
        end = min(video_duration, start + subtitle_duration) if video_duration > 0 else start + subtitle_duration
        if end <= start:
            end = start + subtitle_duration
        text = _ass_escape(_decision_text(item))
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Panel,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _render_episode(
    *,
    video_dir: Path,
    output_dir: Path,
    episode_idx: int,
    episode_decisions: list[dict],
    fps: float,
    subtitle_duration: float,
    panel_width: int,
    panel_height: int,
) -> Path:
    input_video = video_dir / f"episode_{episode_idx:06d}" / "front_camera.mp4"
    if not input_video.exists():
        raise FileNotFoundError(f"Missing front camera video: {input_video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"episode_{episode_idx:06d}_front_cosmos_panel.mp4"
    duration = _probe_duration(input_video)

    with tempfile.TemporaryDirectory() as tmp:
        ass_path = Path(tmp) / f"episode_{episode_idx:06d}.ass"
        _write_ass(ass_path, episode_decisions, fps=fps, subtitle_duration=subtitle_duration, video_duration=duration)
        filter_complex = (
            f"[0:v]scale={panel_width}:{panel_height},setsar=1[left];"
            f"color=c=black:s={panel_width}x{panel_height}:r={fps}:d={max(duration, 0.1):.3f}[right];"
            f"[left][right]hstack=inputs=2,subtitles={ass_path.as_posix()}[v]"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_video),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(output_video),
            ],
            check=True,
        )
    return output_video


def main() -> None:
    args = _parse_args()
    video_dir = args.video_dir.resolve()
    output_dir = args.output_dir or (video_dir / "front_cosmos_panel_videos")
    decisions = _read_decisions(video_dir / "cosmos_decisions.jsonl")
    episode_indices = args.episode if args.episode is not None else sorted(decisions)
    if not episode_indices:
        raise RuntimeError("No episodes to render.")
    for episode_idx in episode_indices:
        out = _render_episode(
            video_dir=video_dir,
            output_dir=output_dir,
            episode_idx=episode_idx,
            episode_decisions=decisions.get(episode_idx, []),
            fps=args.fps,
            subtitle_duration=args.subtitle_duration,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
