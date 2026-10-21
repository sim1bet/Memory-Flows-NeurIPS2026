"""Transition, stability-loss, and terminal-plateau measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .activations import existence_threshold, stability_threshold
from .theory import theoretical_escape_time


def first_upward_crossing(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    if values[0] >= 0.0:
        return 0.0
    indices = np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))
    if indices.size == 0:
        return np.nan
    index = int(indices[0])
    return float(index - values[index] / (values[index + 1] - values[index]))


def interpolate(series: np.ndarray, fractional_index: float) -> float:
    if not np.isfinite(fractional_index):
        return np.nan
    left = int(np.floor(fractional_index))
    if left >= len(series) - 1:
        return float(series[-1])
    fraction = fractional_index - left
    return float(series[left] + fraction * (series[left + 1] - series[left]))


def measure_escape_events(
    saliency: np.ndarray,
    overlaps: np.ndarray,
    adjacency: np.ndarray,
    sample_dt_slow: float,
    tau_slow: float,
    kappa: float,
    activation: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
    hold_time: float = 0.01,
) -> pd.DataFrame:
    """Measure stability loss and the later graph-consistent switch."""
    z_abs = np.abs(saliency)
    dominant = np.argmax(overlaps, axis=0)
    hold_steps = max(1, int(round(hold_time / sample_dt_slow)))
    entries: list[tuple[int, int]] = [(0, int(dominant[0]))]
    for index in np.flatnonzero(dominant[1:] != dominant[:-1]) + 1:
        previous, current = int(dominant[index - 1]), int(dominant[index])
        if adjacency[current, previous] == 0.0:
            continue
        if index + hold_steps > dominant.size:
            continue
        if np.all(dominant[index : index + hold_steps] == current):
            if entries[-1][1] != current:
                entries.append((int(index), current))

    threshold = existence_threshold(gain_phi, gain_w)
    records: list[dict[str, float | int]] = []
    for event, (entry_index, source) in enumerate(entries):
        targets = np.flatnonzero(adjacency[:, source])
        if targets.size != 1:
            continue
        successor = int(targets[0])
        next_index = entries[event + 1][0] if event + 1 < len(entries) else saliency.shape[1] - 1
        next_memory = entries[event + 1][1] if event + 1 < len(entries) else -1
        entry = float(z_abs[source, entry_index])
        successor_entry = float(z_abs[successor, entry_index])
        if entry <= threshold:
            continue
        t_max = float(np.log(entry / threshold))
        stop = min(
            next_index,
            entry_index + int(np.ceil(t_max / sample_dt_slow)) + 2,
            saliency.shape[1] - 1,
        )
        source_segment = z_abs[source, entry_index : stop + 1]
        successor_segment = z_abs[successor, entry_index : stop + 1]
        boundary = stability_threshold(source_segment, activation, gain_phi, gain_w)
        local = first_upward_crossing(successor_segment - boundary)
        if np.isfinite(local):
            empirical = local * sample_dt_slow
            absolute = entry_index + local
            z_plus_empirical = interpolate(z_abs[successor], absolute)
            source_at_escape = interpolate(z_abs[source], absolute)
        else:
            empirical = z_plus_empirical = source_at_escape = np.nan

        theory, z_plus_theory, _ = theoretical_escape_time(
            entry,
            kappa,
            activation,
            gain_phi,
            gain_w,
            successor_initial=0.0,
        )
        theory_measured, _, _ = theoretical_escape_time(
            entry,
            kappa,
            activation,
            gain_phi,
            gain_w,
            successor_initial=successor_entry,
        )
        switch = (
            (next_index - entry_index) * sample_dt_slow
            if next_memory == successor
            else np.nan
        )
        records.append(
            {
                "event": event,
                "source": source,
                "successor": successor,
                "entry_time_slow": entry_index * sample_dt_slow,
                "absolute_escape_time_slow": entry_index * sample_dt_slow + empirical,
                "Z_entry": entry,
                "z_successor_at_entry": successor_entry,
                "T_max": t_max,
                "T_esc_empirical": empirical,
                "T_esc_theory": theory,
                "T_esc_theory_measured_initial": theory_measured,
                "T_esc_relative_error_percent": (
                    100.0 * (empirical - theory) / theory
                    if np.isfinite(empirical) and np.isfinite(theory)
                    else np.nan
                ),
                "T_esc_empirical_physical": empirical * tau_slow,
                "T_esc_theory_physical": theory * tau_slow,
                "Z_plus_empirical": z_plus_empirical,
                "Z_plus_theory": z_plus_theory,
                "z_source_at_escape": source_at_escape,
                "T_switch_empirical": switch,
                "switch_delay_after_stability_loss": (
                    switch - empirical
                    if np.isfinite(switch) and np.isfinite(empirical)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class Plateau:
    mean: float
    std: float
    sem: float
    n_events: int
    window_start: float
    window_end: float
    relative_range: float
    relative_drift: float
    slope: float
    ok: bool


def terminal_plateau(
    events: pd.DataFrame,
    window_time: float,
    min_events: int,
    relative_tolerance: float,
) -> Plateau:
    valid = events.dropna(subset=["T_esc_empirical", "absolute_escape_time_slow"]).sort_values(
        "absolute_escape_time_slow"
    )
    if valid.empty:
        return Plateau(*(np.nan,) * 3, 0, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    end = float(valid["absolute_escape_time_slow"].iloc[-1])
    start = end - window_time
    selected = valid[valid["absolute_escape_time_slow"] >= start]
    if len(selected) < min_events:
        selected = valid.tail(min_events)
        start = float(selected["absolute_escape_time_slow"].iloc[0])
    values = selected["T_esc_empirical"].to_numpy(float)
    times = selected["absolute_escape_time_slow"].to_numpy(float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sem = float(std / np.sqrt(values.size))
    relative_range = float(np.ptp(values) / abs(mean)) if mean else np.inf
    if values.size > 1 and np.ptp(times) > 0.0:
        slope = float(np.polyfit(times, values, 1)[0])
        relative_drift = float(abs(slope) * np.ptp(times) / abs(mean))
    else:
        slope = relative_drift = 0.0
    ok = bool(
        values.size >= min_events
        and relative_range <= relative_tolerance
        and relative_drift <= relative_tolerance
    )
    return Plateau(
        mean, std, sem, int(values.size), start, end,
        relative_range, relative_drift, slope, ok
    )
