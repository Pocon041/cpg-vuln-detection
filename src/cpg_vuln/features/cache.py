from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureCacheMetadata:
    rows: int
    dim: int
    dtype: str


class MemmapFeatureCache:
    def __init__(self, root: Path, metadata: FeatureCacheMetadata, *, read_only: bool) -> None:
        self.root = root
        self.metadata = metadata
        mode = "r" if read_only else "r+"
        self.vectors = np.memmap(
            root / "vectors.dat",
            dtype=metadata.dtype,
            mode=mode,
            shape=(metadata.rows, metadata.dim),
        )
        self.completed = np.load(root / "completed.npy", mmap_mode=mode)

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        rows: int,
        dim: int,
        dtype: str = "float16",
    ) -> "MemmapFeatureCache":
        root.mkdir(parents=True, exist_ok=True)
        metadata = FeatureCacheMetadata(rows=rows, dim=dim, dtype=np.dtype(dtype).name)
        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            existing = cls.open(root)
            if existing.metadata != metadata:
                raise ValueError(f"cache shape mismatch at {root}")
            return existing
        metadata_path.write_text(
            json.dumps(asdict(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        vectors = np.memmap(
            root / "vectors.dat",
            dtype=metadata.dtype,
            mode="w+",
            shape=(rows, dim),
        )
        vectors[:] = 0
        vectors.flush()
        np.save(root / "completed.npy", np.zeros(rows, dtype=np.bool_))
        return cls(root, metadata, read_only=False)

    @classmethod
    def open(cls, root: Path, *, read_only: bool = False) -> "MemmapFeatureCache":
        metadata = FeatureCacheMetadata(
            **json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        )
        return cls(root, metadata, read_only=read_only)

    def __len__(self) -> int:
        return self.metadata.rows

    def write(self, indices: list[int], values: np.ndarray) -> None:
        if values.shape != (len(indices), self.metadata.dim):
            raise ValueError("feature batch has an unexpected shape")
        self.vectors[indices] = values.astype(self.metadata.dtype, copy=False)
        self.vectors.flush()
        self.completed[indices] = True
        self.completed.flush()

    def read(self, indices: list[int] | np.ndarray) -> np.ndarray:
        return np.asarray(self.vectors[indices]).copy()

    def pending_indices(self) -> list[int]:
        return np.flatnonzero(~np.asarray(self.completed)).tolist()

    @property
    def is_complete(self) -> bool:
        return bool(np.all(self.completed))

