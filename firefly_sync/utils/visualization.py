"""Visualisation utilities for synchronisation simulation results.

Provides plotting functions for phase evolution, order parameter,
drone positions, and animated flash events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Deferred matplotlib import — only imported when a plot function is called,
# so the module can be imported in headless environments.
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore


def plot_phases(
    phase_history: np.ndarray,
    timestep: float = 0.01,
    title: str = "Oscillator Phase Evolution",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the phase of each oscillator over time.

    Args:
        phase_history: Array of shape (T, N) — phases in radians.
        timestep: Simulation dt for the time axis.
        title: Plot title.
        save_path: If set, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        Matplotlib Figure.
    """
    t, n = phase_history.shape
    time = np.arange(t) * timestep

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n):
        ax.plot(time, phase_history[:, i], label=f"Drone {i}", alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase θ (rad)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_order_parameter(
    phase_history: np.ndarray,
    timestep: float = 0.01,
    title: str = "Kuramoto Order Parameter r(t)",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the order parameter r(t) over the course of a simulation.

    Args:
        phase_history: Array of shape (T, N).
        timestep: Simulation dt for the time axis.
        title: Plot title.
        save_path: If set, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        Matplotlib Figure.
    """
    from firefly_sync.logging.metrics import SynchronizationMetrics

    r_t = SynchronizationMetrics.order_parameter_over_time(phase_history)
    time = np.arange(len(r_t)) * timestep

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, r_t, "b-", linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full sync")
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Sync threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Order Parameter r(t)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_drone_positions(
    positions: list[tuple[float, float]],
    firing_mask: list[bool] | None = None,
    title: str = "Drone Positions",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot drone positions in 2D space.

    Args:
        positions: List of (x, y) coordinates.
        firing_mask: If provided, highlight firing drones in a different
            colour.
        title: Plot title.
        save_path: File path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure.
    """
    positions = np.asarray(positions)
    fig, ax = plt.subplots(figsize=(6, 6))

    if firing_mask is None:
        firing_mask = [False] * len(positions)

    for i, (x, y) in enumerate(positions):
        color = "orange" if firing_mask[i] else "blue"
        marker = "s" if firing_mask[i] else "o"
        size = 150 if firing_mask[i] else 100
        ax.scatter(x, y, c=color, marker=marker, s=size, zorder=5)
        ax.annotate(f"D{i}", (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def animate_flashes(
    phase_history: np.ndarray,
    positions: list[tuple[float, float]],
    timestep: float = 0.01,
    frame_skip: int = 10,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Create an animation of drone positions with flash indicators.

    Flashing drones are shown as larger orange squares; non-flashing
    drones as blue circles.

    Args:
        phase_history: Array of shape (T, N).
        positions: Per-drone (x, y) positions.
        timestep: Simulation dt in seconds.
        frame_skip: Only render every Nth frame.
        save_path: If set, save animation to this path (e.g., '.mp4' or '.gif').

    Returns:
        Matplotlib FuncAnimation object.
    """
    # TODO: implement full animation logic
    # For now, this is a skeleton that returns a placeholder.
    positions = np.asarray(positions)
    t, n = phase_history.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    scat = ax.scatter(positions[:, 0], positions[:, 1], s=100, c="blue")

    def _update(_frame: int) -> tuple:
        # Placeholder update function
        return (scat,)

    _ = FuncAnimation(fig, _update, frames=t // frame_skip, interval=50, blit=True)

    if save_path:
        # TODO: save animation with ffmpeg or pillow writer
        pass

    return _


# ======================================================================
# Step 3A — Batch evaluation plotting functions
# ======================================================================

def plot_timing_error_by_trial(
    trial_data: list[dict],
    title: str = "Follower Flash Timing Error by Trial",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot signed timing error for each follower flash, grouped by trial.

    Each trial dict must contain ``flash_events_csv`` (path to the
    trial's flash_events.csv) and a ``label`` for the legend.

    Parameters
    ----------
    trial_data:
        List of ``{"flash_events_csv": Path, "label": str}`` dicts.
    title:
        Plot title.
    save_path:
        If set, save figure to this path.
    show:
        Whether to call plt.show().

    Returns
    -------
    Matplotlib Figure.
    """
    import csv

    fig, ax = plt.subplots(figsize=(12, 5))

    for td in trial_data:
        path = Path(td["flash_events_csv"])
        if not path.exists():
            continue

        flash_idx: list[int] = []
        errors: list[float] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            idx = 0
            for row in reader:
                if row.get("event_type") == "follower_flash" and row.get("timing_error_s"):
                    try:
                        errors.append(float(row["timing_error_s"]))
                        flash_idx.append(idx)
                        idx += 1
                    except (ValueError, KeyError):
                        pass

        if errors:
            ax.plot(flash_idx, errors, marker=".", markersize=3,
                    linewidth=0.8, alpha=0.7, label=td.get("label", "?"))

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Follower Flash Index")
    ax.set_ylabel("Timing Error (s)  [leader − follower]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_phase_error_by_trial(
    trial_data: list[dict],
    title: str = "Phase Error by Trial",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot wrapped phase error over time, grouped by trial.

    Each trial dict must contain ``oscillator_log_csv`` (path) and
    ``label`` (str).
    """
    import csv

    fig, ax = plt.subplots(figsize=(12, 5))

    for td in trial_data:
        path = Path(td["oscillator_log_csv"])
        if not path.exists():
            continue

        times: list[float] = []
        errors: list[float] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    times.append(float(row["t"]))
                    errors.append(float(row["phase_error_rad"]))
                except (ValueError, KeyError):
                    pass

        if errors:
            ax.plot(times, errors, linewidth=0.5, alpha=0.7,
                    label=td.get("label", "?"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase Error (rad)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_time_to_sync_by_condition(
    summary_rows: list[dict],
    title: str = "Time to Synchronisation by Condition",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of mean time-to-sync ± std for each follower frequency.

    Parameters
    ----------
    summary_rows:
        List of dicts from ``summary_by_condition.csv`` with keys
        ``follower_initial_freq_hz``, ``mean_time_to_sync_s``,
        ``std_time_to_sync_s``.
    """
    freqs = [str(r.get("follower_initial_freq_hz", "?")) for r in summary_rows]
    means = [float(r.get("mean_time_to_sync_s", 0) or 0) for r in summary_rows]
    stds  = [float(r.get("std_time_to_sync_s", 0) or 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(freqs))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color="steelblue", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Time to Sync (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_steady_state_error_by_condition(
    summary_rows: list[dict],
    title: str = "Steady-State Timing Error by Condition",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of steady-state MAE ± std by follower frequency."""
    freqs = [str(r.get("follower_initial_freq_hz", "?")) for r in summary_rows]
    means = [float(r.get("mean_steady_state_mae_s", 0) or 0) for r in summary_rows]
    stds  = [float(r.get("std_steady_state_mae_s", 0) or 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(freqs))
    ax.bar(x, means, yerr=stds, capsize=8, color="darkorange", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Mean Abs Timing Error (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_success_rate_by_condition(
    summary_rows: list[dict],
    title: str = "Synchronisation Success Rate by Condition",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of sync success rate (0–1) by follower frequency."""
    freqs = [str(r.get("follower_initial_freq_hz", "?")) for r in summary_rows]
    rates = [float(r.get("success_rate", 0) or 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(freqs))
    ax.bar(x, rates, color="seagreen", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
