"""Episode-scoped observation memory and immutable, provenance-bearing snapshots."""
from __future__ import annotations

import copy
import json
import re
from collections import OrderedDict
from pathlib import Path

from .evidence_adapters import _json_value


class EpisodeMemory:
    def __init__(self, max_frames=64, retrieval_frames=8, max_objects=128):
        if not 1 <= retrieval_frames <= max_frames or max_objects < 1:
            raise ValueError("invalid memory limits")
        self.max_frames = max_frames
        self.retrieval_frames = retrieval_frames
        self.max_objects = max_objects
        self.reset("uninitialized")

    def reset(self, episode_id):
        self.episode_id = str(episode_id)
        self.frames = OrderedDict()
        self.objects = OrderedDict()
        self.actions = []
        self.version = 0
        self.last_step = -1
        self.last_snapshot = None

    @staticmethod
    def tokens(text):
        return set(re.findall(r"\w+", str(text).lower()))

    def retrieve(self, question):
        query = self.tokens(question)
        ranked = sorted(self.frames.values(), key=lambda frame: (
            len(query & self.tokens(json.dumps(frame["observations"], ensure_ascii=False))),
            frame["step"],
        ), reverse=True)
        selected = ranked[:self.retrieval_frames]
        # Always retain the current observation, even when lexical retrieval disagrees.
        latest = next(reversed(self.frames.values()), None)
        if latest and latest not in selected:
            selected = selected[:self.retrieval_frames - 1] + [latest]
        return sorted(selected, key=lambda frame: frame["step"])

    def update(self, state, image_path=None):
        state = _json_value(copy.deepcopy(state))
        step = int(state.get("step", 0))
        if step < self.last_step:
            raise ValueError("episode step decreased; reset memory before a new episode")
        if step == self.last_step:
            return copy.deepcopy(self.last_snapshot)
        frame_id = f"{self.episode_id}:{step}"
        path = Path(image_path).resolve() if image_path else None
        if path is not None and not path.is_file():
            raise ValueError(f"missing observation image: {path}")
        observations = {"objects": state.get("objects", []), "perception": state.get("perception", {})}
        self.frames[frame_id] = {
            "id": frame_id, "step": step, "image": str(path) if path else None,
            "observations": observations,
        }
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)
        for obj in state.get("objects", []):
            if "id" not in obj:
                continue
            key = str(obj["id"])
            previous = self.objects.get(key, {})
            sources = list(dict.fromkeys(previous.get("frame_ids", []) + [frame_id]))
            conflicts = list(previous.get("conflicts", []))
            if previous.get("category") and previous["category"] != obj.get("category"):
                conflicts.append({"previous_category": previous["category"], "frame_id": frame_id})
            self.objects[key] = dict(obj, frame_ids=sources[-8:], last_seen=step,
                                     source="observed", conflicts=conflicts[-4:])
            self.objects.move_to_end(key)
        while len(self.objects) > self.max_objects:
            self.objects.popitem(last=False)
        self.version += 1
        self.last_step = step
        retrieved = self.retrieve(state["task"]["question"])
        state["state_version"] = self.version
        state["episode_id"] = self.episode_id
        state["memory"] = {"objects": list(self.objects.values()), "frames": retrieved,
                           "actions": self.actions[-16:]}
        state["evidence"] = {
            "observed_frame_ids": [frame["id"] for frame in retrieved],
            "claims": [{"source": "observed", "frame_id": frame["id"],
                        "content": frame["observations"]} for frame in retrieved],
            "coverage_status": "unknown",
            "instruction": "Unobserved regions remain unknown. Repeated views do not imply complete counts or absence.",
        }
        history = state.get("history", {})
        state["budget"] = {"remaining_steps": max(0, int(history.get("max_steps", 0)) - step)}
        images = [frame["image"] for frame in retrieved if frame["image"]]
        if images:
            state["images"] = images
        self.last_snapshot = copy.deepcopy(state)
        return copy.deepcopy(state)

    def record(self, decision):
        self.actions.append({"state_version": self.version, "action": decision.action,
                             "target_id": decision.target_id, "fallback": decision.fallback})
        self.actions = self.actions[-16:]

    def answer_state(self, state):
        result = copy.deepcopy(dict(state))
        frames = self.retrieve(result["task"]["question"])
        images = [frame["image"] for frame in frames if frame["image"]]
        if images:
            result["images"] = images
        return result
