"""Extract policy test pairs (observation sent, chunk emitted) from eval recordings.

CUPID attributes closed-loop outcomes to training data through the policy's own
rollout samples: every chunk request is one test pair ``(s', a')`` where ``s'`` is
the observation the server received and ``a'`` is the full predicted chunk it
returned. ``/policy_observation_sent`` carries the state and camera stamps of
each request, ``/policy_action_chunk`` the loaded chunks, joined on
``capture_stamp == chunk_target_stamp``. Camera pixels are decoded from the
recorded compressed video at those stamps and brought to the policy input with
the station's cover-resize and centre crop.
"""

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

OBS_TOPIC = "/policy_observation_sent"
CHUNK_TOPIC = "/policy_action_chunk"
#: Policy camera key -> recorded colour stream; wire slot rgb_<x> is camera cam_<x>.
CAMERA_TOPICS = {
    "observation.images.rgb_static": "/cam_static/color/image_rect_compressed",
    "observation.images.rgb_left": "/cam_left/color/image_rect_compressed",
    "observation.images.rgb_right": "/cam_right/color/image_rect_compressed",
}
#: Nearest recorded frame accepted when the exact capture stamp is absent.
MAX_FRAME_OFFSET_NS = 40_000_000
SUCCESS_RATINGS = ("excellent", "ok")
FAILURE_RATINGS = ("failed",)


@dataclasses.dataclass
class RolloutPairs:
    """Test pairs of one rollout, in request order."""

    episode_id: int
    placement_combination_id: int | None
    success: bool
    capture_ns: np.ndarray  # (N,)
    state: np.ndarray  # (N, 16) physical train order
    action: np.ndarray  # (N, T, 16) physical train order
    images: dict[str, np.ndarray]  # key -> (N, H, W, 3) uint8 RGB
    dropped: int = 0  # matched requests without a usable camera frame


def _ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose(pose) -> list[float]:
    p, q = pose.position, pose.orientation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def cartesian_action_to_train(action) -> np.ndarray:
    """Wire ``PolicyCartesianAction`` -> train order [pose_l, grip_l, pose_r, grip_r]."""
    return np.asarray(
        [
            *_pose(action.left_arm_pose),
            action.left_gripper_width,
            *_pose(action.right_arm_pose),
            action.right_gripper_width,
        ],
        dtype=np.float64,
    )


def observation_state_to_train(obs) -> np.ndarray:
    """Sent state -> train order, as serving folds the toolkit sub-features."""
    if len(obs.tcp_left) != 1 or len(obs.tcp_right) != 1:
        raise ValueError("observation lacks a measured TCP pose")
    return np.asarray(
        [
            *_pose(obs.tcp_left[0]),
            obs.gripper_left_width,
            *_pose(obs.tcp_right[0]),
            obs.gripper_right_width,
        ],
        dtype=np.float64,
    )


def match_requests(observations: list, chunks: list) -> list[tuple]:
    """Join loaded chunks to the requesting observation by capture stamp.

    A chunk whose grid origin is not a request's capture stamp (cold start or a
    zero target) cannot be attributed to its request reliably and is skipped.
    """
    requested = {_ns(o.capture_stamp): o for o in observations if o.chunk_requested}
    pairs = []
    for chunk in chunks:
        obs = requested.get(_ns(chunk.chunk_target_stamp))
        if obs is not None:
            pairs.append((obs, chunk))
    pairs.sort(key=lambda pair: _ns(pair[0].capture_stamp))
    return pairs


def cover_crop_box(native_h: int, native_w: int, height: int, width: int):
    """Delivery geometry of the station (toolkit ``resize_crop_calibration``).

    Scale until both requested sides are covered, truncating the scaled size
    per axis, then centre-crop whole pixels (the odd pixel of an uneven
    overhang stays on the right/bottom). Returns (scaled_h, scaled_w, top, left).
    """
    scale = max(width / native_w, height / native_h)
    scaled_w, scaled_h = int(native_w * scale), int(native_h * scale)
    return scaled_h, scaled_w, (scaled_h - height) // 2, (scaled_w - width) // 2


def _resize_jpeg(rgb: np.ndarray, height: int, width: int, jpeg_quality: int | None):
    import cv2

    if rgb.shape[:2] != (height, width):
        scaled_h, scaled_w, top, left = cover_crop_box(*rgb.shape[:2], height, width)
        rgb = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        rgb = rgb[top : top + height, left : left + width]
    if jpeg_quality is not None:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def decode_frames_at(
    packets: list[tuple[int, str, bytes]],
    wanted: set[int],
    height: int,
    width: int,
    jpeg_quality: int | None,
) -> dict[int, np.ndarray]:
    """Decode a recorded stream in order, keeping frames nearest to ``wanted`` stamps.

    Every packet is decoded because inter-coded frames depend on predecessors.
    Returns recorded-frame stamp -> resized RGB for each frame that is the nearest
    one (within MAX_FRAME_OFFSET_NS) to some wanted stamp.
    """
    import av

    if not packets:
        return {}
    stamps = np.asarray([stamp for stamp, _, _ in packets], dtype=np.int64)
    keep = set()
    for stamp in wanted:
        index = int(np.argmin(np.abs(stamps - stamp)))
        if abs(int(stamps[index]) - stamp) <= MAX_FRAME_OFFSET_NS:
            keep.add(int(stamps[index]))
    formats = {fmt for _, fmt, _ in packets}
    if len(formats) != 1:
        raise ValueError(f"mixed video formats in one stream: {formats}")
    codec_name = {"av1": "libdav1d", "h264": "h264", "h265": "hevc", "vp9": "vp9"}[
        formats.pop()
    ]
    codec = av.CodecContext.create(codec_name, "r")
    out: dict[int, np.ndarray] = {}

    def take(frames):
        for frame in frames:
            stamp = int(stamps[frame.pts])
            if stamp in keep:
                out[stamp] = _resize_jpeg(
                    frame.to_ndarray(format="rgb24"), height, width, jpeg_quality
                )

    for index, (_, _, data) in enumerate(packets):
        packet = av.Packet(data)
        packet.pts = index
        packet.dts = index
        try:
            take(codec.decode(packet))
        except av.error.InvalidDataError:
            # A recording can start mid-GOP: inter frames before the first
            # keyframe/sequence header are undecodable and carry no request.
            continue
    take(codec.decode(None))
    return out


def read_rollout(
    mcap_path: Path,
    image_shapes: dict[str, tuple[int, int]],
    *,
    episode_id: int,
    success: bool,
    placement_combination_id: int | None = None,
    jpeg_quality: int | None = 85,
) -> RolloutPairs:
    """Read the test pairs of one recorded rollout.

    ``image_shapes`` maps each policy camera key to its (height, width) input.
    Images are cover-resized, centre-cropped and JPEG round-tripped like the
    station's policy stream.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    topics = {CAMERA_TOPICS[key]: key for key in image_shapes}
    observations, chunks = [], []
    packets: dict[str, list] = {key: [] for key in image_shapes}
    with open(mcap_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(
            topics=[OBS_TOPIC, CHUNK_TOPIC, *topics]
        ):
            if channel.topic == OBS_TOPIC:
                observations.append(message)
            elif channel.topic == CHUNK_TOPIC:
                chunks.append(message)
            else:
                packets[topics[channel.topic]].append(
                    (_ns(message.timestamp), message.format, bytes(message.data))
                )
    pairs = match_requests(observations, chunks)
    frames = {}
    for key, (height, width) in image_shapes.items():
        slot = key.removeprefix("observation.images.")
        wanted = {_ns(getattr(obs, f"{slot}_stamp")) for obs, _ in pairs}
        decoded = decode_frames_at(packets[key], wanted, height, width, jpeg_quality)
        frames[key] = (np.asarray(sorted(decoded), dtype=np.int64), decoded)

    capture, states, actions = [], [], []
    images = {key: [] for key in image_shapes}
    dropped = 0
    for obs, chunk in pairs:
        chosen = {}
        for key in image_shapes:
            slot = key.removeprefix("observation.images.")
            stamp = _ns(getattr(obs, f"{slot}_stamp"))
            available, decoded = frames[key]
            if len(available) == 0:
                break
            nearest = int(available[np.argmin(np.abs(available - stamp))])
            if abs(nearest - stamp) > MAX_FRAME_OFFSET_NS:
                break
            chosen[key] = decoded[nearest]
        if len(chosen) != len(image_shapes):
            dropped += 1
            continue
        capture.append(_ns(obs.capture_stamp))
        states.append(observation_state_to_train(obs))
        actions.append(np.stack([cartesian_action_to_train(a) for a in chunk.actions]))
        for key, image in chosen.items():
            images[key].append(image)
    horizon = len(pairs[0][1].actions) if pairs else 0
    return RolloutPairs(
        episode_id=episode_id,
        placement_combination_id=placement_combination_id,
        success=success,
        capture_ns=np.asarray(capture, dtype=np.int64),
        state=np.asarray(states, dtype=np.float64).reshape(-1, 16),
        action=np.asarray(actions, dtype=np.float64).reshape(-1, horizon, 16),
        images={
            key: np.asarray(value, dtype=np.uint8).reshape(-1, *image_shapes[key], 3)
            for key, value in images.items()
        },
        dropped=dropped,
    )


def rollout_outcomes(eval_dir: Path) -> list[dict]:
    """Episodes counted in the eval's selected SR with a binary outcome.

    CUPID uses R(tau) = +1 for success and -1 for failure (paper A.2); success is
    an excellent/ok rating as in the platform's SR. Uncounted (ignored,
    unrated or intervention) attempts carry no reviewed outcome and are skipped.
    """
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    rows = []
    for entry in manifest["episodes"]:
        episode = entry["episode"]
        if not entry.get("counted_in_selected_sr"):
            continue
        rating = episode.get("rating")
        if rating in SUCCESS_RATINGS:
            success = True
        elif rating in FAILURE_RATINGS:
            success = False
        else:
            continue
        rows.append(
            {
                "episode_id": int(episode["id"]),
                "placement_combination_id": episode.get("placement_combination_id"),
                "success": success,
                "path": eval_dir / entry["file"],
            }
        )
    return rows


def iter_rollouts(
    eval_dir: Path, image_shapes: dict[str, tuple[int, int]], **kwargs
) -> Iterator[RolloutPairs]:
    """Stream counted rollouts of one eval download in manifest order."""
    for row in rollout_outcomes(eval_dir):
        yield read_rollout(
            row["path"],
            image_shapes,
            episode_id=row["episode_id"],
            success=row["success"],
            placement_combination_id=row["placement_combination_id"],
            **kwargs,
        )
