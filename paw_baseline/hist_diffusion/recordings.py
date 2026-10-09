"""Inspect eval MCAP recordings: channels, policy chunk cadence and execution.

Usage:
  python -m paw_baseline.hist_diffusion.recordings summary <file.mcap>
  python -m paw_baseline.hist_diffusion.recordings attachment <file.mcap> <name>
  python -m paw_baseline.hist_diffusion.recordings sample <file.mcap> <topic> [n]
  python -m paw_baseline.hist_diffusion.recordings cadence <file.mcap>...
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def summary(path: Path) -> dict:
    """Channel topics, encodings, schema names and message counts."""
    with path.open("rb") as stream:
        reader = make_reader(stream)
        info = reader.get_summary()
        channels = {
            cid: {
                "topic": ch.topic,
                "encoding": ch.message_encoding,
                "schema": info.schemas[ch.schema_id].name if ch.schema_id else None,
            }
            for cid, ch in info.channels.items()
        }
        counts = Counter()
        if info.statistics is not None:
            counts.update(info.statistics.channel_message_counts)
        attachments = [a.name for a in (info.attachment_indexes or [])]
    return {
        "channels": [
            {**meta, "messages": counts.get(cid, 0)}
            for cid, meta in sorted(channels.items(), key=lambda kv: kv[1]["topic"])
        ],
        "attachments": attachments,
    }


def attachment(path: Path, name: str) -> str:
    with path.open("rb") as stream:
        for item in make_reader(stream).iter_attachments():
            if item.name == name:
                return item.data.decode()
    raise KeyError(name)


def _to_plain(value, depth=0):
    """ROS message → plain Python for printing."""
    if depth > 6:
        return "..."
    if hasattr(value, "__slots__"):
        return {k: _to_plain(getattr(value, k), depth + 1) for k in value.__slots__}
    if isinstance(value, (list, tuple)):
        if len(value) > 6:
            return [_to_plain(v, depth + 1) for v in value[:3]] + [f"... ({len(value)})"]
        return [_to_plain(v, depth + 1) for v in value]
    return value


def messages(path: Path, topics: list[str]):
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        yield from reader.iter_decoded_messages(topics=topics)


def sample(path: Path, topic: str, n: int) -> list:
    out = []
    for _, channel, message, decoded in messages(path, [topic]):
        out.append({"log_time": message.log_time, "msg": _to_plain(decoded)})
        if len(out) >= n:
            break
    return out


SLOT_NS = 1e9 / 30


def _stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def cadence(path: Path) -> dict:
    """How the station executed the policy's chunks in one episode.

    plan_interval_slots: spacing of successive chunk groundings (the server's
    replanning interval). executed_slots_per_chunk: dispatched slots taken
    from each chunk before a later one superseded it (open-loop execution).
    first_slot_index: index of a chunk's first dispatched slot (stale slots
    dropped for latency). in_blend_fraction: dispatches inside a crossfade.
    """
    targets, arrivals, lengths = [], [], []
    for _, _, message, chunk in messages(path, ["/policy_action_chunk"]):
        targets.append(_stamp_ns(chunk.chunk_target_stamp))
        arrivals.append(message.log_time)
        lengths.append(len(chunk.actions))
    executed: Counter = Counter()
    first_slot: dict[int, int] = {}
    in_blend = dispatched = 0
    round_trips = []
    for _, _, _, status in messages(path, ["/policy_action_status"]):
        if not status.dispatched_action:
            continue
        dispatched += 1
        number = status.current_chunk_number
        executed[number] += 1
        first_slot.setdefault(number, status.slot_index)
        in_blend += bool(status.in_blend)
        if status.last_chunk_wait_ms > 0:
            round_trips.append(status.last_chunk_wait_ms)
    intervals = np.diff(np.asarray(targets)) / SLOT_NS if len(targets) > 1 else np.zeros(1)
    # Ignore the episode's final chunk, which runs until the episode ends.
    per_chunk = [executed[n] for n in sorted(executed)[:-1]]

    def stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return None
        return {
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }

    return {
        "chunks": len(targets),
        "chunk_length": stats(lengths),
        "dispatched_slots": dispatched,
        "plan_interval_slots": stats(intervals),
        "executed_slots_per_chunk": stats(per_chunk),
        "first_slot_index": stats(list(first_slot.values())),
        "arrival_minus_target_ms": stats((np.asarray(arrivals) - np.asarray(targets)) / 1e6),
        "chunk_wait_ms": stats(round_trips),
        "in_blend_fraction": in_blend / max(dispatched, 1),
    }


def seams(path: Path) -> dict:
    """Commanded TCP step sizes (mm per slot) at chunk switches vs within chunks."""
    previous_chunk, previous_pos = None, None
    at_switch, within, jumps = [], [], []
    for _, _, _, status in messages(path, ["/policy_action_status"]):
        if not status.dispatched_action:
            continue
        pose = status.dispatched
        pos = np.array(
            [
                [pose.left_arm_pose.position.x, pose.left_arm_pose.position.y, pose.left_arm_pose.position.z],
                [pose.right_arm_pose.position.x, pose.right_arm_pose.position.y, pose.right_arm_pose.position.z],
            ]
        )
        if previous_pos is not None:
            step = 1000 * float(np.linalg.norm(pos - previous_pos, axis=1).max())
            (at_switch if status.current_chunk_number != previous_chunk else within).append(step)
        jumps.append(status.last_tcp_jump_mm)
        previous_chunk, previous_pos = status.current_chunk_number, pos

    def stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return None
        return {
            "n": int(values.size),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    return {
        "step_mm_at_switch": stats(at_switch),
        "step_mm_within_chunk": stats(within),
        "last_tcp_jump_mm": stats(jumps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["summary", "attachment", "sample", "cadence", "seams"]
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("arg", nargs="?")
    parser.add_argument("n", nargs="?", type=int, default=2)
    args = parser.parse_args()
    if args.command == "summary":
        print(json.dumps(summary(args.path), indent=1))
    elif args.command == "attachment":
        print(attachment(args.path, args.arg))
    elif args.command == "cadence":
        print(json.dumps(cadence(args.path)))
    elif args.command == "seams":
        print(json.dumps(seams(args.path)))
    else:
        print(json.dumps(sample(args.path, args.arg, args.n), indent=1, default=str))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
