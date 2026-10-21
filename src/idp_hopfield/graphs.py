"""Directed memory-transition graph helpers."""

from __future__ import annotations

import numpy as np


def sequence_adjacency(n_memories: int, sequence: list[int] | tuple[int, ...]) -> np.ndarray:
    adjacency = np.zeros((n_memories, n_memories), dtype=float)
    for source, target in zip(sequence[:-1], sequence[1:]):
        adjacency[int(target), int(source)] = 1.0
    return adjacency


def cycle_adjacency(n_memories: int, order: list[int] | np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    if order.size < 2 or np.unique(order).size != order.size:
        raise ValueError("A cycle needs at least two distinct nodes.")
    adjacency = np.zeros((n_memories, n_memories), dtype=float)
    adjacency[order[(np.arange(order.size) + 1) % order.size], order] = 1.0
    return adjacency


def successor_array(adjacency: np.ndarray) -> np.ndarray:
    successors = np.full(adjacency.shape[0], -1, dtype=int)
    for source in range(adjacency.shape[1]):
        targets = np.flatnonzero(adjacency[:, source])
        if targets.size == 1:
            successors[source] = int(targets[0])
    return successors


def scaled_cycle_length(n_memories: int, fraction: float = 1.0 / 3.0) -> int:
    return int(np.clip(round(float(fraction) * n_memories), 2, n_memories))


def nested_fanout_successors(
    rng: np.random.Generator,
    n_memories: int,
    source: int,
    maximum_outdegree: int,
) -> np.ndarray:
    candidates = np.delete(np.arange(n_memories), source)
    if maximum_outdegree > candidates.size:
        raise ValueError("Out-degree exceeds P-1.")
    return rng.permutation(candidates)[:maximum_outdegree]
