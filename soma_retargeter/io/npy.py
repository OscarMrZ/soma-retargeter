# SPDX-License-Identifier: Apache-2.0

"""Export retargeted motions as a SuperMimic motion library.

Output folder (all clips concatenated along time):
  qpos.npy   (N, 7 + nj) float32  root pos (m), root quat (wxyz), joints (rad, slides in m)
  qvel.npy   (N, 6 + nj) float32  MuJoCo free-joint convention: root linear velocity (world),
                                  root angular velocity (body frame), joint velocities
  clips.json {"fps", "joint_names", "clips": [{"name", "category", "start", "length"}]}
"""

import json
import pathlib

import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


def buffer_to_qpos_qvel(buffer: CSVAnimationBuffer, fps: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a ``CSVAnimationBuffer`` to ``fps`` and return its MuJoCo-layout state.

    Args:
        buffer (CSVAnimationBuffer): Frames of ``[root pos, root quat (xyzw), joints]``.
        fps (float, optional): Output frame rate. Defaults to 50.0.

    Returns:
        tuple: ``qpos`` (T, 7 + nj) and finite-difference ``qvel`` (T, 6 + nj), float64.

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
    # one-sided at the ends, expressed in the body frame (MuJoCo free-joint qvel).
    frames = np.arange(num_frames)
    prev_frames = np.maximum(frames - 1, 0)
    next_frames = np.minimum(frames + 1, num_frames - 1)
    span = ((next_frames - prev_frames) * dt)[:, None]
    ang_vel_world = (root_rot[next_frames] * root_rot[prev_frames].inv()).as_rotvec() / span
    ang_vel_body = root_rot.inv().apply(ang_vel_world)

    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    qvel = np.concatenate(
        [np.gradient(root_pos, dt, axis=0), ang_vel_body, np.gradient(joint_pos, dt, axis=0)], axis=1)
    return qpos, qvel


def save_library(
    out_dir: str,
    joint_names: list[str],
    clips: list[tuple[str, str, CSVAnimationBuffer]],
    fps: float = 50.0,
) -> None:
    """
    Save retargeted motions as one SuperMimic motion library folder.

    Args:
        out_dir (str): Folder receiving ``qpos.npy``, ``qvel.npy`` and ``clips.json``.
        joint_names (list): Names of the joint columns, in ``joint_q`` order (root excluded).
        clips (list): ``(name, category, buffer)`` for every motion to pack.
        fps (float, optional): Frame rate of the library. Defaults to 50.0.

    Raises:
        RuntimeError: If there are no clips or a buffer does not match ``joint_names``.
        OSError: If the files cannot be written.
    """
    if len(clips) == 0:
        raise RuntimeError("[ERROR]: No motions to export.")

    qpos_clips, qvel_clips, clips_meta = [], [], []
    start = 0
    for name, category, buffer in sorted(clips, key=lambda clip: clip[0]):
        num_joints = np.shape(buffer.data)[1] - 7
        if num_joints != len(joint_names):
            raise RuntimeError(
                f"[ERROR]: Motion [{name}] has {num_joints} joint values, expected {len(joint_names)}.")

        qpos, qvel = buffer_to_qpos_qvel(buffer, fps)
        qpos_clips.append(qpos.astype(np.float32))
        qvel_clips.append(qvel.astype(np.float32))
        clips_meta.append({"name": name, "category": category, "start": start, "length": len(qpos)})
        start += len(qpos)

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "qpos.npy", np.concatenate(qpos_clips))
    np.save(out_path / "qvel.npy", np.concatenate(qvel_clips))

    meta = {"fps": fps, "joint_names": list(joint_names), "source": "soma-retargeter", "clips": clips_meta}
    (out_path / "clips.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[INFO]: Saved {len(clips_meta)} motions, {start} frames at {fps} fps to [{out_path}]")
