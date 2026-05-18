from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


TASK_OVERLAY_LABELS = {
    1: "Left trocar pickup",
    2: "Right trocar pickup",
    3: "Align trocars",
    4: "Insert trocar",
    5: "Placement",
}


def task_overlay_label(task_number: int, retry: bool = False) -> str:
    label = TASK_OVERLAY_LABELS.get(task_number, f"Task {task_number}")
    return f"{label} RETRY" if retry else label


def retry_reasons(ep: dict) -> list[str]:
    tasks = ep.get("tasks", [])
    nums = [t.get("task_number") for t in tasks]
    reasons = []
    if len(nums) != len(set(nums)):
        reasons.append("duplicate_task_records")
    if int(ep.get("recover_count", 0)) > 0:
        reasons.append("episode_recover_count")
    if any(int(t.get("attempts", 0)) > 1 for t in tasks):
        reasons.append("task_attempts_gt_1")
    if any(int(t.get("recover_count", 0)) > 0 for t in tasks):
        reasons.append("task_recover_count")
    return reasons


def category_for(ep: dict) -> str:
    success = bool(ep.get("success"))
    retried = bool(retry_reasons(ep))
    if success and retried:
        return "success_retry"
    if success:
        return "success_no_retry"
    if retried:
        return "failure_retry"
    return "failure_no_retry"


def split_attempt_steps(total_steps: int, attempts: int) -> list[int]:
    if attempts <= 1:
        return [max(total_steps, 0)]
    steps = []
    remaining = max(total_steps, 0)
    for _ in range(attempts - 1):
        step = min(60, remaining)
        steps.append(step)
        remaining -= step
    steps.append(max(remaining, 0))
    return steps


def label_segments(ep: dict, fps: float) -> list[tuple[float, float, str]]:
    frame_cursor = 0
    segments = []

    def add(frames: int, label: str) -> None:
        nonlocal frame_cursor
        frames = max(int(frames), 1)
        start = frame_cursor / fps
        end = (frame_cursor + frames) / fps
        segments.append((start, end, label))
        frame_cursor += frames

    add(1, "Episode start")
    previous_counts: dict[int, int] = {}
    for task in ep.get("tasks", []):
        task_number = int(task.get("task_number"))
        attempts = int(task.get("attempts", 1))
        total_steps = int(task.get("policy_steps", 0))
        recover_count = int(task.get("recover_count", 0))
        occurrence = previous_counts.get(task_number, 0)
        previous_counts[task_number] = occurrence + 1

        for attempt_idx, steps in enumerate(split_attempt_steps(total_steps, attempts), start=1):
            task4_retry_context = (task_number == 3 and occurrence > 0) or (
                task_number == 4 and occurrence > 0
            )
            retry = attempt_idx > 1 or occurrence > 0
            label_task_number = 4 if task4_retry_context else task_number
            label = task_overlay_label(label_task_number, retry or task4_retry_context)
            add(steps, label)
            if task_number == 3 and attempt_idx < attempts:
                add(60, task_overlay_label(4 if task4_retry_context else 3, retry=True))
            if task_number == 4 and attempt_idx < attempts:
                add(60, task_overlay_label(4, retry=True))

        if task_number == 4 and not bool(task.get("success")):
            add(30, task_overlay_label(4, retry=True))

    return segments


def ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def esc_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_ass(path: Path, segments: list[tuple[float, float, str]], duration: float) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 480
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,0,0,0,0,100,100,0,0,3,1,0,7,16,16,16,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for start, end, text in segments:
        if start >= duration:
            break
        end = min(max(end, start + 0.03), duration)
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{esc_ass(text)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def probe(src: Path) -> tuple[float, float]:
    data = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration",
            "-of",
            "json",
            str(src),
        ],
        text=True,
    )
    stream = json.loads(data)["streams"][0]
    rate = stream.get("avg_frame_rate", "30/1")
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(stream.get("duration") or 0.0)
    return fps or 30.0, duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()

    base = args.base
    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"front_camera_fourway_overlay_review_{stamp}"
    categories = ["success_retry", "success_no_retry", "failure_retry", "failure_no_retry"]
    for cat in categories:
        (out_dir / cat).mkdir(parents=True, exist_ok=False)
    ass_dir = out_dir / "_ass"
    ass_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    missing = []
    counts = {cat: 0 for cat in categories}
    for ep in summary.get("episodes", []):
        ep_idx = int(ep["episode_idx"])
        src = base / f"episode_{ep_idx:06d}" / "front_camera.mp4"
        category = category_for(ep)
        dst = out_dir / category / f"episode_{ep_idx:06d}_front_camera_overlay.mp4"
        if not src.exists():
            missing.append(str(src))
            continue

        fps, duration = probe(src)
        ass_path = ass_dir / f"episode_{ep_idx:06d}.ass"
        write_ass(ass_path, label_segments(ep, fps), duration)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                f"ass={ass_path}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-an",
                str(dst),
            ],
            check=True,
        )
        reasons = retry_reasons(ep)
        counts[category] += 1
        manifest.append(
            {
                "episode_idx": ep_idx,
                "category": category,
                "success": bool(ep.get("success")),
                "retry_triggered": bool(reasons),
                "retry_reasons": reasons,
                "source": str(src),
                "copied_to": str(dst),
            }
        )
        if len(manifest) % 10 == 0:
            print(f"processed {len(manifest)} videos", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "summary_source": str(summary_path),
                "counts": counts,
                "missing": missing,
                "note": "Overlay labels are reconstructed from summary task order, policy_steps, attempts, and recover counts.",
                "episodes": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(ass_dir)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir.parent))

    print(f"OUT_DIR={out_dir}")
    print(f"ZIP={zip_path}")
    for cat in categories:
        print(f"{cat.upper()}={counts[cat]}")
    print(f"MISSING={len(missing)}")


if __name__ == "__main__":
    main()
