import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.audio_embedding import (
    compute_embedding_from_mono,
    cosine_similarity,
    load_audio_clip_pcm,
    pcm_s16le_to_mono_float,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioProfile:
    id: str
    name: str
    tags: List[str]
    embedding: List[float]
    created_at: str
    updated_at: str
    clip_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tags": list(self.tags),
            "embedding": list(self.embedding),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clip_id": self.clip_id,
        }


class AudioProfileStore:
    """
    Stores user-tagged audio profiles in `data/audio_profiles.json`.
    Profiles are matched using cosine similarity on lightweight embeddings.
    """

    def __init__(self, store_path: str = "data/audio_profiles.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Dict[str, Any]:
        try:
            if not self.store_path.exists():
                return {"profiles": []}
            return json.loads(self.store_path.read_text())
        except Exception:
            return {"profiles": []}

    def _save_raw(self, obj: Dict[str, Any]) -> None:
        tmp = self.store_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(str(tmp), str(self.store_path))

    def list_profiles(self, include_embedding: bool = False) -> List[Dict[str, Any]]:
        raw = self._load_raw()
        out: List[Dict[str, Any]] = []
        for p in raw.get("profiles", []) or []:
            if not isinstance(p, dict):
                continue
            d = dict(p)
            if not include_embedding:
                d.pop("embedding", None)
            out.append(d)
        # newest first
        out.sort(key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""), reverse=True)
        return out

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        raw = self._load_raw()
        for p in raw.get("profiles", []) or []:
            if isinstance(p, dict) and p.get("id") == profile_id:
                return dict(p)
        return None

    def delete_profile(self, profile_id: str) -> bool:
        raw = self._load_raw()
        profiles = [p for p in (raw.get("profiles", []) or []) if not (isinstance(p, dict) and p.get("id") == profile_id)]
        if len(profiles) == len(raw.get("profiles", []) or []):
            return False
        raw["profiles"] = profiles
        self._save_raw(raw)
        return True

    def create_profile_from_clip(
        self,
        *,
        name: str,
        tags: Optional[List[str]],
        clip_path: str,
        clip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        nm = str(name).strip() or "Audio Profile"
        tg = [t.strip() for t in (tags or []) if str(t).strip()]
        tg = list(dict.fromkeys(tg))

        clip = load_audio_clip_pcm(clip_path)
        x = pcm_s16le_to_mono_float(clip.pcm_s16le, clip.channels)
        emb = compute_embedding_from_mono(x, sample_rate=clip.sample_rate)

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        prof = AudioProfile(
            id=f"ap_{uuid.uuid4().hex[:12]}",
            name=nm,
            tags=tg,
            embedding=emb,
            created_at=now,
            updated_at=now,
            clip_id=clip_id,
        )

        raw = self._load_raw()
        profiles = list(raw.get("profiles", []) or [])
        profiles.insert(0, prof.to_dict())
        raw["profiles"] = profiles
        self._save_raw(raw)
        return prof.to_dict()

    def match_embedding(
        self,
        embedding: List[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.65,
    ) -> List[Dict[str, Any]]:
        raw = self._load_raw()
        results: List[Tuple[float, Dict[str, Any]]] = []
        for p in raw.get("profiles", []) or []:
            if not isinstance(p, dict):
                continue
            emb = p.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            try:
                sim = cosine_similarity(embedding, emb)
            except Exception:
                continue
            if sim >= float(min_similarity):
                results.append((float(sim), dict(p)))
        results.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for sim, p in results[: int(max(1, top_k))]:
            p2 = dict(p)
            p2.pop("embedding", None)
            p2["similarity"] = float(sim)
            out.append(p2)
        return out


