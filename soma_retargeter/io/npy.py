# SPDX-License-Identifier: Apache-2.0

"""Export retargeted motions as a SuperMimic motion library.

Output folder (all clips concatenated along time, one float32 file per field):
  root_pos_w.npy      (N, 3)   root position (m)
  root_quat_w.npy     (N, 4)   root orientation (wxyz)
  root_lin_vel_w.npy  (N, 3)   root linear velocity (world frame)
  root_ang_vel_w.npy  (N, 3)   root angular velocity (world frame)
  joint_pos.npy       (N, nj)  joints (rad, slides in m)
  joint_vel.npy       (N, nj)  joint velocities
  motion_ends.npy     (M,) int64  end frame (exclusive) of each motion
  motion_names.json, joint_names.json
"""

import json
import pathlib

import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


def buffer_to_fields(buffer: CSVAnimationBuffer, fps: float = 50.0) -> dict[str, np.ndarray]:
    """
    Resample a ``CSVAnimationBuffer`` to ``fps`` and return its SuperMimic motion fields.

    Args:
        buffer (CSVAnimationBuffer): Frames of ``[root pos, root quat (xyzw), joints]``.
        fps (float, optional): Output frame rate. Defaults to 50.0.

    Returns:
        dict: The six fields of the module docstring, (T, k) float64, velocities by finite differences.

    Raises:
        RuntimeError: If the buffer is too short to resample and differentiate.
    """
    if buffer is None or buffer.num_frames < 2:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    data = np.asarray(buffer.data, dtype=np.float64)
    src_time = np.arange(len(data)) / buffer.sample_rate
    num_frames = int(np.floor(src_time[-1] * fps + 1e-6)) + 1
    if num_frames < 2:
        raise RuntimeError(f"[ERROR]: Buffer too short to export at {fps} fps.")
    time = np.minimum(np.arange(num_frames) / fps, src_time[-1])
    dt = 1.0 / fps

    def resample(values):
        return np.stack([np.interp(time, src_time, column) for column in values.T], axis=1)

    root_pos = resample(data[:, 0:3])
    joint_pos = resample(data[:, 7:])
    root_rot = Slerp(src_time, R.from_quat(data[:, 3:7]))(time)

    # xyzw -> wxyz, with hemisphere continuity so neighbouring quaternions are close.
    root_quat = root_rot.as_quat()[:, [3, 0, 1, 2]]
    flips = np.where(np.sum(root_quat[1:] * root_quat[:-1], axis=1) < 0.0, -1.0, 1.0)
    root_quat[1:] *= np.cumprod(flips)[:, None]

    # Angular velocity: central differences of the world-frame rotation increment,
    # one-sided at the ends.
    frames = np.arange(num_frames)
    prev_frames = np.maximum(frames - 1, 0)
    next_frames = np.minimum(frames + 1, num_frames - 1)
    span = ((next_frames - prev_frames) * dt)[:, None]
    ang_vel_world = (root_rot[next_frames] * root_rot[prev_frames].inv()).as_rotvec() / span

    return {
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_w": np.gradient(root_pos, dt, axis=0),
        "root_ang_vel_w": ang_vel_world,
        "joint_pos": joint_pos,
        "joint_vel": np.gradient(joint_pos, dt, axis=0),
    }


def save_library(
    out_dir: str,
    joint_names: list[str],
    clips: list[tuple[str, CSVAnimationBuffer]],
    fps: float = 50.0,
) -> None:
    """
    Save retargeted motions as one SuperMimic motion library folder.

    Args:
        out_dir (str): Folder receiving the files of the module docstring.
        joint_names (list): Names of the joint columns, in ``joint_q`` order (root excluded).
        clips (list): ``(name, buffer)`` for every motion to pack.
        fps (float, optional): Frame rate of the library. Defaults to 50.0.

    Raises:
        RuntimeError: If there are no clips or a buffer does not match ``joint_names``.
        OSError: If the files cannot be written.
    """
    if len(clips) == 0:
        raise RuntimeError("[ERROR]: No motions to export.")

    field_clips, names, lengths = {}, [], []
    for name, buffer in sorted(clips, key=lambda clip: clip[0]):
        num_joints = np.shape(buffer.data)[1] - 7
        if num_joints != len(joint_names):
            raise RuntimeError(
                f"[ERROR]: Motion [{name}] has {num_joints} joint values, expected {len(joint_names)}.")

        for field, values in buffer_to_fields(buffer, fps).items():
            field_clips.setdefault(field, []).append(values.astype(np.float32))
        names.append(name)
        lengths.append(len(values))

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for field, values in field_clips.items():
        np.save(out_path / f"{field}.npy", np.concatenate(values))
    (out_path / "motion_names.json").write_text(json.dumps(names), encoding="utf-8")
    (out_path / "joint_names.json").write_text(json.dumps(list(joint_names)), encoding="utf-8")
    # Written last: its presence marks the folder as complete.
    np.save(out_path / "motion_ends.npy", np.cumsum(lengths, dtype=np.int64))
    print(f"[INFO]: Saved {len(names)} motions, {sum(lengths)} frames at {fps} fps to [{out_path}]")
