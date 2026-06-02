from __future__ import annotations

import random
from collections import defaultdict

from cpg_vuln.data.audit import ManifestRecord


def stratified_split(records: list[ManifestRecord], *, seed: int) -> dict[str, list[str]]:
    by_label: dict[int, list[str]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record.sample_id)
    result = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        items = sorted(by_label[label])
        random.Random(seed + label).shuffle(items)
        train_count, val_count = _split_sizes(len(items))
        result["train"].extend(items[:train_count])
        result["val"].extend(items[train_count : train_count + val_count])
        result["test"].extend(items[train_count + val_count :])
    return _sorted_splits(result)


def grouped_stratified_split(
    records: list[ManifestRecord], *, seed: int
) -> dict[str, list[str]]:
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        groups[record.source_hash].append(record)
    by_label: dict[int, list[list[ManifestRecord]]] = defaultdict(list)
    for group in groups.values():
        labels = {record.label for record in group}
        if len(labels) != 1:
            raise ValueError("source hash group contains conflicting labels")
        by_label[next(iter(labels))].append(group)
    result = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        label_groups = sorted(by_label[label], key=lambda group: group[0].source_hash)
        random.Random(seed + label).shuffle(label_groups)
        train_count, val_count = _split_sizes(len(label_groups))
        partitions = (
            ("train", label_groups[:train_count]),
            ("val", label_groups[train_count : train_count + val_count]),
            ("test", label_groups[train_count + val_count :]),
        )
        for split_name, partition in partitions:
            result[split_name].extend(
                record.sample_id for group in partition for record in group
            )
    return _sorted_splits(result)


def _split_sizes(size: int) -> tuple[int, int]:
    train = round(size * 0.8)
    validation = round(size * 0.1)
    if train + validation > size:
        validation = max(0, size - train)
    return train, validation


def _sorted_splits(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: sorted(sample_ids) for name, sample_ids in splits.items()}

