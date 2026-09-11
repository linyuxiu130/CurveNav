import requests
import numpy as np
import io
import json
import os
import threading

from navbench.depth import as_depth_meters

_THREAD_LOCAL = threading.local()
_DEFAULT_TIMEOUT = float(os.environ.get("NAVBENCH_HTTP_TIMEOUT", "120"))


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Convert camera output to contiguous uint8 for the raw RGB tensor.

    Isaac Lab cameras often output float32 in [0,1], while the wire contract is uint8.
    """
    arr = np.asarray(img)
    # Normalize channels: JPEG expects 1 or 3 channels; drop alpha if present.
    if arr.ndim >= 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Heuristic: scale [0,1] -> [0,255]
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(arr)
    # Integer types (including uint16): clamp to [0,255].
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _prepare_depth(depth: np.ndarray) -> np.ndarray:
    """Return lossless float32 depth in meters for the one wire protocol."""
    return as_depth_meters(depth)


def _array_file(name: str, array: np.ndarray):
    """Create one multipart item from contiguous raw tensor bytes."""
    arr = np.ascontiguousarray(array)
    return (f"{name}.raw", memoryview(arr).cast("B"), "application/octet-stream")


def _add_raw_metadata(data: dict, name: str, array: np.ndarray) -> None:
    data[f"{name}_shape"] = json.dumps(list(array.shape))
    data[f"{name}_dtype"] = array.dtype.str


def _prepare_rgb(images) -> np.ndarray:
    return _to_uint8_image(np.asarray(images))


def _decode_policy_response(response):
    response.raise_for_status()
    if response.headers.get("Content-Type", "").split(";", 1)[0] != "application/x-npz":
        raise ValueError("policy response must be application/x-npz")
    with np.load(io.BytesIO(response.content), allow_pickle=False) as payload:
        trajectory = payload["trajectory"]
    return trajectory, None, None


def _post(url, *, files=None, data=None, json_payload=None):
    if not hasattr(_THREAD_LOCAL, "session"):
        _THREAD_LOCAL.session = requests.Session()
    return _THREAD_LOCAL.session.post(
        url, files=files, data=data, json=json_payload, timeout=_DEFAULT_TIMEOUT
    )


def navigator_reset(
    intrinsic=None,
    stop_threshold=-3.0,
    batch_size=1,
    port=8888,
    env_id=None,
    seed=1234,
    sample_indices=(),
    sample_idx=None,
    scene_name="",
    global_batch_size=None,
    batch_start=0,
):
    if env_id is None:
        url = "http://localhost:%d/navigator_reset" % port
        payload = {
            "intrinsic": intrinsic.tolist(),
            "stop_threshold": stop_threshold,
            "batch_size": batch_size,
            "global_batch_size": (
                batch_size if global_batch_size is None else global_batch_size
            ),
            "batch_start": batch_start,
            "seed": seed,
        }
        payload["sample_indices"] = list(sample_indices)
        payload["scene_name"] = scene_name
        response = _post(url, json_payload=payload)
    else:
        url = "http://localhost:%d/navigator_reset_env" % port
        payload = {"env_id": env_id}
        payload["sample_idx"] = sample_idx
        payload["scene_name"] = scene_name
        response = _post(url, json_payload=payload)
    response.raise_for_status()
    return response.json()


def navigator_shutdown(port=8888, timeout=3.0):
    """End one policy-server lifetime and its caller-side HTTP session."""
    requests.post(
        f"http://localhost:{port}/shutdown", json={}, timeout=timeout
    ).raise_for_status()
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is not None:
        session.close()
        del _THREAD_LOCAL.session


def pointgoal_step(
    pointgoal,
    rgb,
    depth,
    robot_pos,
    robot_quat,
    body_to_world,
    camera_to_body,
    camera_intrinsics,
    timestamps,
    planning_goal,
    port=8888,
    **rgbd_context,
):
    """Call a policy server with the argument names used by X-NavDP upstream."""
    url = "http://localhost:%d/pointgoal_step" % port
    arrays = rgbd_context if rgbd_context else {
        "image": _prepare_rgb(rgb), "depth": _prepare_depth(depth)
    }
    files = {name: _array_file(name, value) for name, value in arrays.items()}
    data = {
        "goal_data": json.dumps(
            {"goal_x": pointgoal[:, 0].tolist(), "goal_y": pointgoal[:, 1].tolist()}
        ),
    }
    data["state_data"] = json.dumps(
        {
            "robot_pos": np.asarray(robot_pos).tolist(),
            "robot_quat": np.asarray(robot_quat).tolist(),
            "body_to_world": np.asarray(body_to_world).tolist(),
            "camera_to_body": np.asarray(camera_to_body).tolist(),
            "camera_intrinsics": np.asarray(camera_intrinsics).tolist(),
            "timestamps": np.asarray(timestamps).tolist(),
            "planning_goal": np.asarray(planning_goal).tolist(),
        }
    )
    for name, value in arrays.items():
        _add_raw_metadata(data, name, value)
    return _decode_policy_response(_post(url, files=files, data=data))
