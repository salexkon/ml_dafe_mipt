import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

import copy
import os
import time
from pathlib import Path

from .week11_utils import (
    ImageDataset, benchmark_inference, channel_statistics, choose_device,
    classification_metrics, evaluate_model, make_loader, paired_accuracy_interval,
    plot_histories, predict_probabilities, seed_everything, synchronize,
)

# Одна поверхность для всех наглядных примеров: минимум в (0,0).
# rotation меняет ориентацию осей, но сохраняет собственные значения 1 и 20.
def optimizer_objective(point, rotation=0.0):
    if point.ndim == 2:
        point = point.diagonal()
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    first = cosine * point[0] + sine * point[1]
    second = -sine * point[0] + cosine * point[1]
    value = (first.square() + 20 * second.square()) / 2
    return value


def trace_solver(point, solver, steps=100, rotation=0.0, noise_std=0.0, seed=10):
    generator = np.random.default_rng(seed)
    gradient_noise = generator.normal(0, noise_std, size=(steps, 2))
    coordinates = point.diagonal() if point.ndim == 2 else point
    positions = [coordinates.detach().numpy().copy()]
    losses = [optimizer_objective(point, rotation).item()]
    update_norms = []
    state_history = []
    started = time.perf_counter()
    for step in range(steps):
        solver.zero_grad(set_to_none=True)
        loss = optimizer_objective(point, rotation)
        noise = torch.tensor(gradient_noise[step], dtype=point.dtype)
        coordinates = point.diagonal() if point.ndim == 2 else point
        surrogate = loss + (noise * coordinates).sum()
        surrogate.backward()
        previous = point.detach().clone()
        solver.step()
        update_norms.append((point.detach() - previous).norm().item())
        coordinates = point.diagonal() if point.ndim == 2 else point
        positions.append(coordinates.detach().numpy().copy())
        losses.append(optimizer_objective(point, rotation).item())
        state = solver.state.get(point, {})
        copied = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                copied[key] = value.detach().numpy().copy()
        state_history.append(copied)
        if not np.isfinite(losses[-1]) or losses[-1] > 1e12:
            break
    result = {"positions": np.asarray(positions), "losses": np.asarray(losses),
              "update_norms": np.asarray(update_norms), "states": state_history,
              "rotation": rotation, "seconds": time.perf_counter() - started}
    return result


def plot_solver_trace(trace, name):
    positions = trace["positions"]
    horizontal = np.linspace(-3.5, 3.5, 140)
    vertical = np.linspace(-2.5, 2.5, 140)
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    cosine = np.cos(trace["rotation"])
    sine = np.sin(trace["rotation"])
    first = cosine * grid_x + sine * grid_y
    second = -sine * grid_x + cosine * grid_y
    surface = (first ** 2 + 20 * second ** 2) / 2
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    contours = axes[0].contour(grid_x, grid_y, surface, levels=[.05, .2, .5, 1, 2, 5, 10, 20, 40], colors="#bdcbd9", linewidths=.8)
    axes[0].clabel(contours, levels=[.5, 2, 10, 40], inline=True, fontsize=7)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].plot(positions[:, 0], positions[:, 1], marker=".", markersize=3, linewidth=1)
    axes[0].scatter(*positions[0], color="darkred", marker="x", s=70, label="старт")
    axes[0].scatter(0, 0, color="black", marker="*", s=90, label="минимум")
    axes[0].set(xlabel="w0", ylabel="w1", title=name)
    axes[0].legend(fontsize=8)
    axes[1].semilogy(np.maximum(trace["losses"], 1e-14))
    axes[1].set(xlabel="шаг", ylabel="f(w), log", title="Значение общей функции")
    axes[2].plot(trace["update_norms"])
    axes[2].set_ylim(0, max(float(np.max(trace["update_norms"])) * 1.1, 1e-12))
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].set(xlabel="шаг", ylabel="||Deltaw||", title="Фактическая длина обновления")
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    plt.show()

solver_traces = {}
