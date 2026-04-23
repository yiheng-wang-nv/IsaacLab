# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI runner for the multi-task trocar GR00T policy.

This script opens Isaac Sim with the trocar environment and lets the user:

* press ``1``-``5`` to select one of the split-stage task prompts,
* press ``SPACE`` to toggle continuous policy inference,
* press ``S`` to execute a single policy step,
* press ``R`` to reset the environment.

It is intended for debugging multi-task prompt switching and stage transitions
with a finetuned Isaac-GR00T checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an interactive GUI for the multi-task trocar policy.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the Isaac-GR00T checkpoint directory.")
parser.add_argument(
    "--task_id",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0",
    help="Gym task id for the trocar environment.",
)
parser.add_argument("--denoising_steps", type=int, default=4, help="GR00T denoising steps.")
parser.add_argument("--step_hz", type=float, default=10.0, help="Continuous inference stepping rate.")
parser.add_argument(
    "--open_loop_steps",
    type=int,
    default=8,
    help="Number of env steps to execute before running policy inference again.",
)
parser.add_argument(
    "--task_timeout_steps",
    type=int,
    default=60,
    help="Pause continuous inference when the selected task has executed this many steps.",
)
parser.add_argument(
    "--model_device",
    type=str,
    default=None,
    help="Device for the policy. Defaults to the simulator device when omitted.",
)
parser.add_argument("--seed", type=int, default=None, help="Optional environment reset seed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True


def _infer_display_from_x11_sockets() -> str | None:
    x11_dir = Path("/tmp/.X11-unix")
    if not x11_dir.exists():
        return None
    display_ids: list[int] = []
    for socket_path in x11_dir.glob("X*"):
        suffix = socket_path.name[1:]
        if suffix.isdigit():
            display_ids.append(int(suffix))
    if not display_ids:
        return None
    return f":{min(display_ids)}"


if not os.environ.get("DISPLAY"):
    inferred_display = _infer_display_from_x11_sockets()
    if inferred_display is not None:
        os.environ["DISPLAY"] = inferred_display
        print(f"[DEBUG] DISPLAY was unset. Inferred DISPLAY={inferred_display} from /tmp/.X11-unix.")
    else:
        print("[DEBUG] DISPLAY is unset and no X11 socket was found under /tmp/.X11-unix.")

if not os.environ.get("XAUTHORITY"):
    default_xauthority = Path.home() / ".Xauthority"
    if default_xauthority.exists():
        os.environ["XAUTHORITY"] = str(default_xauthority)
        print(f"[DEBUG] XAUTHORITY was unset. Using {default_xauthority}.")

# IsaacLab's AppLauncher will still force headless unless a Kit visualizer is
# explicitly requested. Default to `kit` unless the user already provided a
# visualizer selection on the CLI.
if getattr(args, "visualizer", None) in (None, []):
    args.visualizer = ["kit"]
    args.visualizer_explicit = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[DEBUG] DISPLAY =", os.environ.get("DISPLAY"))
print("[DEBUG] visualizer =", getattr(args, "visualizer", None))
print("[DEBUG] AppLauncher._headless =", getattr(app_launcher, "_headless", None))
print("[DEBUG] AppLauncher._sim_experience_file =", getattr(app_launcher, "_sim_experience_file", None))


import carb
import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.kit.app
import omni.ui as ui
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TASK_DESCRIPTIONS = [
    "left hand pick up trocar",
    "right hand pick up trocar",
    "align trocars",
    "install trocar",
    "place trocar",
]
TASK3_INDEX = 2

CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]

PERM_TO_REF = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    19,
    20,
    15,
    18,
    14,
    17,
    23,
    26,
    27,
    21,
    24,
    22,
    25,
]
INV_PERM = np.argsort(PERM_TO_REF)
BODY_JOINT_INDICES = [
    0,
    3,
    6,
    9,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
]
DEX3_JOINT_INDICES = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
SHOULDER_SLICE = (15, 29)
ACTION_PREFIX_PAD = 15
ROBOT_ACTION_DIM = ACTION_PREFIX_PAD + len(INV_PERM)


def _print_controls() -> None:
    print("Interactive trocar controls:")
    print("  1-5 : select task prompt")
    print("  SPACE / P : toggle continuous inference")
    print("  S : run one policy step")
    print("  R : reset the environment")
    print("  H : print this help again")


def _find_isaac_gr00t_root(model_path: Path) -> Path:
    for candidate in [model_path, *model_path.parents]:
        if (candidate / "gr00t_config.py").exists() and (candidate / "gr00t" / "model" / "policy.py").exists():
            return candidate
    env_root = Path("/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T")
    if (env_root / "gr00t_config.py").exists():
        return env_root
    raise FileNotFoundError(
        "Could not locate Isaac-GR00T root. Set the checkpoint path inside the Isaac-GR00T repo "
        "or update `_find_isaac_gr00t_root`."
    )


def _load_policy(model_path: str, model_device: str, denoising_steps: int):
    model_path_obj = Path(model_path).expanduser().resolve()
    isaac_gr00t_root = _find_isaac_gr00t_root(model_path_obj)
    if str(isaac_gr00t_root) not in sys.path:
        sys.path.insert(0, str(isaac_gr00t_root))

    from gr00t.model.policy import Gr00tPolicy

    spec = importlib.util.spec_from_file_location(
        "interactive_trocar_gr00t_config",
        str(isaac_gr00t_root / "gr00t_config.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load gr00t_config.py from {isaac_gr00t_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_config = module.UnitreeG1SimInferDataConfig()

    policy = Gr00tPolicy(
        model_path=str(model_path_obj),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=denoising_steps,
        device=model_device,
    )
    return policy


def _create_env(task_id: str, device: str):
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=1)
    for cam_name in CAMERA_KEYS:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.data_types = ["rgb"]
    env_cfg.recorders = {}
    return gym.make(task_id, cfg=env_cfg).unwrapped


def _flush_observations(env):
    app = omni.kit.app.get_app()
    env.sim.set_setting("/app/player/playSimulations", False)
    app.update()
    env.sim.set_setting("/app/player/playSimulations", True)
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)
    return env.observation_manager.compute(update_history=True)


def _reset_env(env, seed: int | None):
    obs, _ = env.reset(seed=seed)
    obs = _flush_observations(env)
    return obs


def _get_joint_states(env) -> np.ndarray:
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    device = joint_pos.device

    body_idx = torch.tensor(BODY_JOINT_INDICES, device=device, dtype=torch.long)
    body_pos = joint_pos[:, body_idx]
    shoulder_pos = body_pos[:, SHOULDER_SLICE[0] : SHOULDER_SLICE[1]]

    dex3_idx = torch.tensor(DEX3_JOINT_INDICES, device=device, dtype=torch.long)
    dex3_pos = joint_pos[:, dex3_idx]

    states = torch.cat([shoulder_pos, dex3_pos], dim=-1)
    return states.cpu().numpy().astype(np.float32)


def _get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs.astype(np.uint8)


def _build_policy_obs(env, task_description: str) -> dict[str, np.ndarray | list[str]]:
    states_internal = _get_joint_states(env)[0]
    states_ref = states_internal[PERM_TO_REF]

    obs_dict: dict[str, np.ndarray | list[str]] = {
        "state.left_arm": states_ref[0:7][np.newaxis],
        "state.right_arm": states_ref[7:14][np.newaxis],
        "state.left_hand": states_ref[14:21][np.newaxis],
        "state.right_hand": states_ref[21:28][np.newaxis],
        "annotation.human.task_description": [task_description],
    }

    cam_map = {
        "front_camera": "video.room_view",
        "left_wrist_camera": "video.left_wrist_view",
        "right_wrist_camera": "video.right_wrist_view",
    }
    for cam_key, gr00t_key in cam_map.items():
        obs_dict[gr00t_key] = _get_camera_rgb(env, cam_key)[0][np.newaxis]

    return obs_dict


def _convert_policy_action_chunk_to_env(action_dict: dict) -> list[np.ndarray]:
    parts = []
    for key in ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]:
        value = np.asarray(action_dict[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        elif value.ndim == 1:
            value = value[np.newaxis, :]
        parts.append(value)

    chunk_len = min(part.shape[0] for part in parts)
    action_chunk: list[np.ndarray] = []
    for step_idx in range(chunk_len):
        action_ref = np.concatenate([part[step_idx] for part in parts], axis=-1)
        action_internal = action_ref[INV_PERM]
        action = np.concatenate([np.zeros(ACTION_PREFIX_PAD, dtype=np.float32), action_internal], axis=0)
        action_chunk.append(_normalize_env_raw_action(action))
    return action_chunk


def _extract_stage_prediction(action_dict: dict) -> tuple[int | None, list[float] | None]:
    stage_pred = action_dict.get("stage_pred")
    stage_logits = action_dict.get("stage_logits")

    stage_pred_value: int | None = None
    if stage_pred is not None:
        if isinstance(stage_pred, torch.Tensor):
            stage_pred_value = int(stage_pred.reshape(-1)[0].item())
        else:
            stage_pred_value = int(np.asarray(stage_pred).reshape(-1)[0])

    stage_prob_values: list[float] | None = None
    if stage_logits is not None:
        if isinstance(stage_logits, torch.Tensor):
            logits = stage_logits.reshape(-1).float()
        else:
            logits = torch.tensor(np.asarray(stage_logits).reshape(-1), dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        stage_prob_values = [float(x) for x in probs.tolist()]

    return stage_pred_value, stage_prob_values


def _get_env_stage_info(env) -> tuple[int, int]:
    env_stage = int(env._task_stage[0].item()) if hasattr(env, "_task_stage") else 0
    lift_z = 0.85483 + 0.05
    t1_pos = wp.to_torch(env.scene["trocar_1"].data.root_pos_w)[0]
    t2_pos = wp.to_torch(env.scene["trocar_2"].data.root_pos_w)[0]
    either_lifted = bool((t1_pos[2] > lift_z) or (t2_pos[2] > lift_z))

    dataset_stage = env_stage + 1 if env_stage >= 1 else (1 if either_lifted else 0)
    return env_stage, dataset_stage


def _format_stage_probs(stage_probs: list[float] | None) -> str:
    if not stage_probs:
        return "n/a"
    return ", ".join(f"{i + 1}:{prob:.2f}" for i, prob in enumerate(stage_probs))


def _normalize_env_raw_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32).reshape(ROBOT_ACTION_DIM).copy()
    action_np[:ACTION_PREFIX_PAD] = 0.0
    return action_np


def _raw_action_to_env_tensor(action: np.ndarray, device: str) -> torch.Tensor:
    action = _normalize_env_raw_action(action).reshape(1, ROBOT_ACTION_DIM)
    return torch.tensor(action, dtype=torch.float32, device=device)


class KeyboardInterface:
    """Keyboard-driven control state for the interactive runner."""

    _TASK_KEY_MAP = {
        "1": 0,
        "KEY_1": 0,
        "NUMPAD_1": 0,
        "2": 1,
        "KEY_2": 1,
        "NUMPAD_2": 1,
        "3": 2,
        "KEY_3": 2,
        "NUMPAD_3": 2,
        "4": 3,
        "KEY_4": 3,
        "NUMPAD_4": 3,
        "5": 4,
        "KEY_5": 4,
        "NUMPAD_5": 4,
    }

    def __init__(self):
        self.running = False
        self.pending_single_step = False
        self.pending_reset = False
        self.pending_retry_task3 = False
        self.selected_task_idx = 0
        self.stop_on_predicted_next_stage = True
        self.last_message = "Ready. Select a task and press SPACE."
        self.last_key = "n/a"
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = None if self._app_window is None else self._app_window.get_keyboard()
        self._sub_keyboard = None
        if self._keyboard is not None:
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        else:
            self.last_message = "Keyboard subscription is unavailable. Use the on-screen buttons."

    def close(self):
        if self._sub_keyboard is not None and self._keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def consume_single_step(self) -> bool:
        value = self.pending_single_step
        self.pending_single_step = False
        return value

    def consume_reset(self) -> bool:
        value = self.pending_reset
        self.pending_reset = False
        return value

    def consume_retry_task3(self) -> bool:
        value = self.pending_retry_task3
        self.pending_retry_task3 = False
        return value

    def select_task(self, task_idx: int) -> None:
        self.selected_task_idx = max(0, min(task_idx, len(TASK_DESCRIPTIONS) - 1))
        self.last_message = f"Selected task {self.selected_task_idx + 1}: {TASK_DESCRIPTIONS[self.selected_task_idx]}"

    def toggle_running(self) -> None:
        self.running = not self.running
        state = "running" if self.running else "paused"
        self.last_message = f"Continuous inference {state}."

    def toggle_stop_on_predicted_next_stage(self) -> None:
        self.stop_on_predicted_next_stage = not self.stop_on_predicted_next_stage
        state = "enabled" if self.stop_on_predicted_next_stage else "disabled"
        self.last_message = f"Auto-pause on predicted next stage is {state}."

    def queue_single_step(self) -> None:
        self.pending_single_step = True
        self.last_message = "Queued one policy step."

    def queue_reset(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_reset = True
        self.last_message = "Queued environment reset."

    def queue_retry_task3(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_retry_task3 = True
        self.selected_task_idx = TASK3_INDEX
        self.last_message = "Queued a Task 3 retry motion."

    def print_help(self) -> None:
        _print_controls()
        self.last_message = "Printed controls to the terminal."

    @staticmethod
    def _key_aliases(raw_key: str) -> set[str]:
        key = raw_key.upper()
        aliases = {key}
        for prefix in ("KEY_", "NUMPAD_", "NUMROW_", "KP_"):
            if key.startswith(prefix):
                aliases.add(key[len(prefix) :])
        return aliases

    def _on_keyboard_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        raw_key = str(event.input.name)
        aliases = self._key_aliases(raw_key)
        self.last_key = raw_key

        for alias in aliases:
            if alias in self._TASK_KEY_MAP:
                self.select_task(self._TASK_KEY_MAP[alias])
                return True

        if aliases & {"SPACE", "SPACEBAR", "P"}:
            self.toggle_running()
        elif aliases & {"S", "ENTER"}:
            self.queue_single_step()
        elif aliases & {"R"}:
            self.queue_reset()
        elif aliases & {"H", "F1"}:
            self.print_help()
        return True


class StatusPanel:
    """Simple docked status panel shown inside Isaac Sim."""

    def __init__(self, keyboard: KeyboardInterface):
        self._keyboard = keyboard
        self._window = ui.Window(
            "Trocar Multitask Debug",
            width=420,
            height=360,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                self._status = ui.Label("", word_wrap=True)
                ui.Spacer(height=8)
                ui.Label("Mouse Controls", word_wrap=True)
                with ui.HStack(spacing=4):
                    for task_idx, task_name in enumerate(TASK_DESCRIPTIONS):
                        ui.Button(
                            f"{task_idx + 1}",
                            width=0,
                            clicked_fn=lambda idx=task_idx: self._keyboard.select_task(idx),
                            tooltip=f"Select task {task_idx + 1}: {task_name}",
                        )
                with ui.HStack(spacing=4):
                    ui.Button(
                        "Start/Pause",
                        width=0,
                        clicked_fn=self._keyboard.toggle_running,
                    )
                    ui.Button(
                        "Step",
                        width=0,
                        clicked_fn=self._keyboard.queue_single_step,
                    )
                    ui.Button(
                        "Reset",
                        width=0,
                        clicked_fn=self._keyboard.queue_reset,
                    )
                    ui.Button(
                        "Retry Task 3",
                        width=0,
                        clicked_fn=self._keyboard.queue_retry_task3,
                    )
                    ui.Button(
                        "Help",
                        width=0,
                        clicked_fn=self._keyboard.print_help,
                    )
                with ui.HStack(spacing=4):
                    ui.Button(
                        "Toggle Stage-Pred Stop",
                        width=0,
                        clicked_fn=self._keyboard.toggle_stop_on_predicted_next_stage,
                    )
                ui.Spacer(height=4)
                self._controls = ui.Label(
                    "Keyboard Controls\n"
                    "1-5: select task\n"
                    "SPACE/P: start or pause\n"
                    "S: single step\n"
                    "R: reset\n"
                    "H: print help\n"
                    "\n"
                    "VNC note: click the viewport or this panel before pressing keys.\n"
                    "If keyboard focus is awkward, use the buttons above.",
                    word_wrap=True,
                )

    def update(
        self,
        keyboard: KeyboardInterface,
        selected_task: str,
        env_stage: int,
        dataset_stage: int,
        model_stage_pred: int | None,
        stage_probs: list[float] | None,
        step_count: int,
        episode_done: bool,
        task_run_step_count: int,
        task3_history_len: int,
        retry_replay_active: bool,
        retry_replay_progress: int,
        retry_replay_total: int,
        open_loop_steps_remaining: int,
    ):
        mode = "RUNNING" if keyboard.running else "PAUSED"
        if episode_done:
            mode += " / EPISODE DONE"
        if retry_replay_active:
            mode += " / RETRY REPLAY"

        if model_stage_pred is None:
            model_stage_text = "n/a"
        elif 0 <= model_stage_pred < len(TASK_DESCRIPTIONS):
            model_stage_text = f"{model_stage_pred + 1}: {TASK_DESCRIPTIONS[model_stage_pred]}"
        else:
            model_stage_text = str(model_stage_pred)
        self._status.text = (
            f"Mode: {mode}\n"
            f"Selected task: {keyboard.selected_task_idx + 1}: {selected_task}\n"
            f"Env raw stage: {env_stage}\n"
            f"Dataset-style stage: {dataset_stage}\n"
            f"Model stage prediction: {model_stage_text}\n"
            f"Model stage probs: {_format_stage_probs(stage_probs)}\n"
            f"Stop on predicted next stage: {keyboard.stop_on_predicted_next_stage}\n"
            f"Task run steps: {task_run_step_count} / {args.task_timeout_steps}\n"
            f"Open-loop steps remaining: {open_loop_steps_remaining} / {args.open_loop_steps}\n"
            f"Task 3 recorded actions: {task3_history_len}\n"
            f"Retry replay steps: {retry_replay_progress} / {retry_replay_total}\n"
            f"Episode steps: {step_count}\n"
            f"Last key seen: {keyboard.last_key}\n"
            f"Last event: {keyboard.last_message}"
        )


def main():
    model_device = args.model_device or args.device
    policy = _load_policy(args.model_path, model_device=model_device, denoising_steps=args.denoising_steps)
    env = _create_env(args.task_id, device=args.device)

    keyboard = KeyboardInterface()
    panel = StatusPanel(keyboard)
    _print_controls()

    obs = _reset_env(env, seed=args.seed)
    del obs

    step_count = 0
    episode_done = False
    model_stage_pred: int | None = None
    stage_probs: list[float] | None = None
    last_step_time = 0.0
    step_period = 1.0 / max(args.step_hz, 1e-6)
    task_run_step_count = 0
    previous_running = False
    previous_task_idx = keyboard.selected_task_idx
    task3_action_history: list[np.ndarray] = []
    task3_history_locked = False
    retry_replay_active = False
    retry_replay_actions: list[np.ndarray] = []
    retry_replay_step_count = 0
    cached_policy_actions: list[np.ndarray] = []
    open_loop_steps_remaining = 0

    try:
        while simulation_app.is_running():
            if keyboard.selected_task_idx != previous_task_idx:
                if previous_task_idx == TASK3_INDEX and task3_action_history and not task3_history_locked:
                    task3_history_locked = True
                task_run_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                previous_task_idx = keyboard.selected_task_idx

            if keyboard.running and not previous_running:
                task_run_step_count = 0
            previous_running = keyboard.running

            if keyboard.consume_reset():
                _reset_env(env, seed=args.seed)
                step_count = 0
                episode_done = False
                model_stage_pred = None
                stage_probs = None
                task_run_step_count = 0
                task3_action_history = []
                task3_history_locked = False
                retry_replay_active = False
                retry_replay_actions = []
                retry_replay_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0

            if keyboard.consume_retry_task3():
                if not task3_action_history:
                    keyboard.last_message = "Retry Task 3 requested, but Task 3 has not been executed yet."
                else:
                    keyboard.running = False
                    task3_history_locked = True
                    retry_replay_active = True
                    retry_replay_actions = [action.copy() for action in reversed(task3_action_history)]
                    retry_replay_step_count = 0
                    task_run_step_count = 0
                    model_stage_pred = None
                    stage_probs = None
                    cached_policy_actions = []
                    open_loop_steps_remaining = 0
                    keyboard.last_message = (
                        f"Started Task 3 reverse replay with {len(retry_replay_actions)} recorded actions."
                    )

            should_step = False
            if not episode_done and keyboard.consume_single_step():
                should_step = True
            if not episode_done and keyboard.running:
                now = time.perf_counter()
                if now - last_step_time >= step_period:
                    should_step = True
                    last_step_time = now

            action_tensor: torch.Tensor | None = None
            executing_retry_replay_step = False
            executing_policy_step = False
            executed_policy_action: np.ndarray | None = None
            if retry_replay_active and not episode_done:
                if retry_replay_step_count >= len(retry_replay_actions):
                    retry_replay_active = False
                    retry_replay_actions = []
                    retry_replay_step_count = 0
                    keyboard.last_message = "Finished Task 3 reverse replay. Press Start to run Task 3 again."
                else:
                    executing_retry_replay_step = True
                    action_tensor = _raw_action_to_env_tensor(
                        retry_replay_actions[retry_replay_step_count], device=env.device
                    )

            if should_step and action_tensor is None:
                executing_policy_step = True
                if not cached_policy_actions:
                    task_description = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                    policy_obs = _build_policy_obs(env, task_description)
                    action_dict = policy.get_action(policy_obs)
                    model_stage_pred, stage_probs = _extract_stage_prediction(action_dict)
                    action_chunk = _convert_policy_action_chunk_to_env(action_dict)
                    if not action_chunk:
                        raise RuntimeError("Policy returned an empty action chunk.")
                    cached_policy_actions = [action.copy() for action in action_chunk[: max(args.open_loop_steps, 1)]]
                    open_loop_steps_remaining = len(cached_policy_actions)
                executed_policy_action = cached_policy_actions.pop(0)
                action_tensor = _raw_action_to_env_tensor(executed_policy_action, device=env.device)

            if should_step or executing_retry_replay_step:
                if action_tensor is None:
                    raise RuntimeError("Expected an action tensor before stepping the environment.")
                _, _, terminated, truncated, _ = env.step(action_tensor)
                step_count += 1
                episode_done = bool(terminated[0]) or bool(truncated[0])
                if executing_retry_replay_step:
                    retry_replay_step_count += 1
                    if retry_replay_step_count >= len(retry_replay_actions):
                        retry_replay_active = False
                        retry_replay_actions = []
                        retry_replay_step_count = 0
                        keyboard.last_message = "Finished Task 3 reverse replay. Press Start to run Task 3 again."
                elif executing_policy_step:
                    if keyboard.selected_task_idx == TASK3_INDEX and not task3_history_locked and executed_policy_action is not None:
                        task3_action_history.append(executed_policy_action.copy())
                        if len(task3_action_history) == 1:
                            keyboard.last_message = "Started recording Task 3 action history for retry."
                    open_loop_steps_remaining = len(cached_policy_actions)
                    if keyboard.running:
                        task_run_step_count += 1
                elif keyboard.running:
                    task_run_step_count += 1

                if executing_retry_replay_step:
                    pass
                elif (
                    keyboard.stop_on_predicted_next_stage
                    and model_stage_pred is not None
                    and model_stage_pred >= keyboard.selected_task_idx + 1
                ):
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused by stage_pred: selected task {keyboard.selected_task_idx + 1}, "
                        f"predicted stage {model_stage_pred + 1}."
                    )
                elif keyboard.running and task_run_step_count >= args.task_timeout_steps:
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused by step timeout after {task_run_step_count} steps "
                        f"(limit {args.task_timeout_steps})."
                    )
                if episode_done:
                    keyboard.running = False
                    retry_replay_active = False
                    retry_replay_actions = []
                    retry_replay_step_count = 0
                    cached_policy_actions = []
                    open_loop_steps_remaining = 0
                    reason = "success/drop/termination" if bool(terminated[0]) else "timeout/truncation"
                    keyboard.last_message = f"Episode finished: {reason}. Press R to reset."
            else:
                env.sim.render()
                time.sleep(1.0 / 60.0)

            env_stage, dataset_stage = _get_env_stage_info(env)
            panel.update(
                keyboard=keyboard,
                selected_task=TASK_DESCRIPTIONS[keyboard.selected_task_idx],
                env_stage=env_stage,
                dataset_stage=dataset_stage,
                model_stage_pred=model_stage_pred,
                stage_probs=stage_probs,
                step_count=step_count,
                episode_done=episode_done,
                task_run_step_count=task_run_step_count,
                task3_history_len=len(task3_action_history),
                retry_replay_active=retry_replay_active,
                retry_replay_progress=retry_replay_step_count,
                retry_replay_total=len(retry_replay_actions),
                open_loop_steps_remaining=open_loop_steps_remaining,
            )
    finally:
        keyboard.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
