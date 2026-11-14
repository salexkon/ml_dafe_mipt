"""Данные, измерения и графики занятия 11. Обучение и модели раскрыты в ноутбуках."""
from __future__ import annotations

import copy
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support,
                             top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

CLASSES = ["самолет", "автомобиль", "птица", "кошка", "олень",
           "собака", "лягушка", "лошадь", "корабль", "грузовик"]
SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    requested = os.environ.get("W11_DEVICE")
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_data(train_size=12000, val_size=2000, test_size=2000, seed=SEED):
    """Отдельные стратифицированные подмножества официальных train/test. NHWC uint8."""
    root = Path(os.environ.get("SCIKIT_LEARN_DATA", Path.home() / "scikit_learn_data"))
    folder = root / "cifar-10-batches-py"
    if folder.exists():
        arrays = []
        labels = []
        for number in range(1, 6):
            with (folder / f"data_batch_{number}").open("rb") as stream:
                batch = pickle.load(stream, encoding="bytes")
            arrays.append(batch[b"data"])
            labels.extend(batch[b"labels"])
        flat = np.concatenate(arrays)
        train_images = flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        train_labels = np.asarray(labels, dtype=np.int64)
        with (folder / "test_batch").open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        test_images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        test_labels = np.asarray(batch[b"labels"], dtype=np.int64)
    else:
        train_source = datasets.CIFAR10(root=str(root), train=True, download=True)
        test_source = datasets.CIFAR10(root=str(root), train=False, download=True)
        train_images = train_source.data
        train_labels = np.asarray(train_source.targets, dtype=np.int64)
        test_images = test_source.data
        test_labels = np.asarray(test_source.targets, dtype=np.int64)

    all_indices = np.arange(len(train_labels))
    train_pool, val_indices = train_test_split(
        all_indices, test_size=val_size, stratify=train_labels, random_state=seed)
    if train_size < len(train_pool):
        train_indices, _ = train_test_split(
            train_pool, train_size=train_size, stratify=train_labels[train_pool], random_state=seed)
    else:
        if train_size != len(train_pool):
            raise ValueError("train_size превышает доступную обучающую часть")
        train_indices = train_pool
    test_indices = np.arange(len(test_labels))
    if test_size < len(test_indices):
        test_indices, _ = train_test_split(
            test_indices, train_size=test_size, stratify=test_labels, random_state=seed)
    elif test_size != len(test_indices):
        raise ValueError("test_size превышает официальный тест")

    result = {
        "train_images": train_images[train_indices], "train_labels": train_labels[train_indices],
        "val_images": train_images[val_indices], "val_labels": train_labels[val_indices],
        "test_images": test_images[test_indices], "test_labels": test_labels[test_indices],
        "train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices,
    }
    return result


def channel_statistics(images):
    values = images.astype(np.float32) / 255.0
    mean = values.mean(axis=(0, 1, 2))
    std = values.std(axis=(0, 1, 2))
    return mean, std


class ImageDataset(Dataset):
    """Хранят uint8 CHW; случайное преобразование вызывают заново при каждом чтении."""
    def __init__(self, images, labels, mean, std, augment=False):
        self.images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        steps = []
        if augment:
            steps.extend([v2.RandomCrop(32, padding=4, padding_mode="reflect"),
                          v2.RandomHorizontalFlip(p=0.5),
                          v2.ColorJitter(brightness=0.3, contrast=0.3)])
        steps.extend([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean.tolist(), std.tolist())])
        self.transform = v2.Compose(steps)

    def __len__(self):
        length = len(self.labels)
        return length

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label


def make_loader(dataset, batch_size=128, shuffle=False, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        generator=generator, num_workers=0)
    return loader


@torch.inference_mode()
def predict_probabilities(model, loader, device):
    model.eval()
    parts = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        parts.append(probabilities.cpu().numpy())
    result = np.concatenate(parts)
    return result


def classification_metrics(y, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(10)[y]
    squared_error = (probabilities - one_hot) ** 2
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", labels=np.arange(10), zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "top3_accuracy": top_k_accuracy_score(y, probabilities, k=3, labels=np.arange(10)),
        "log_loss": log_loss(y, probabilities, labels=np.arange(10)),
        "brier": squared_error.sum(axis=1).mean(),
    }
    return metrics


def parameter_count(model):
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return count


@torch.inference_mode()
def benchmark_inference(model, example, device, repeats=30, warmup=10):
    """Только forward: вход уже на устройстве; синхронизация перед/после каждого замера."""
    model.eval()
    example = example.to(device)
    for _ in range(warmup):
        model(example)
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        model(example)
        synchronize(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
    median = float(np.median(times))
    p90 = float(np.quantile(times, 0.9))
    result = {"latency_ms": 1000 * median, "p90_ms": 1000 * p90,
              "images_s": len(example) / median}
    return result


def evaluate_model(name, model, loader, device, train_seconds):
    probabilities = predict_probabilities(model, loader, device)
    labels = loader.dataset.labels.numpy()
    result = classification_metrics(labels, probabilities)
    images, _ = next(iter(loader))
    single = benchmark_inference(model, images[:1], device)
    batch = benchmark_inference(model, images[:128], device)
    result.update({"model": name, "parameters": parameter_count(model),
                   "parameter_MiB": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
                   "train_s": train_seconds, "batch1_ms": single["latency_ms"],
                   "batch1_p90_ms": single["p90_ms"], "batch128_ms": batch["latency_ms"],
                   "batch128_p90_ms": batch["p90_ms"], "images_s": batch["images_s"],
                   "inference_batch": len(images[:128]), "device": str(device)})
    return result, probabilities


def show_table(frame):
    from IPython.display import display
    numeric = frame.select_dtypes(include="number").columns
    style = frame.style.format(precision=4).background_gradient(cmap="Blues", subset=numeric)
    display(style)


def show_images(images, titles=None, columns=6):
    count = len(images)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.4 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(images[index])
            if titles is not None:
                axis.set_title(str(titles[index]), fontsize=9)
    figure.tight_layout()
    plt.show()


def plot_histories(histories):
    from .lesson11_visuals import PALETTE
    count = len(histories)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.7 * rows),
                                squeeze=False, sharex=True, sharey=True, layout="constrained")
    for index, (axis, (name, history)) in enumerate(zip(axes.flat, histories.items())):
        frame = pd.DataFrame(history)
        color = PALETTE[index % len(PALETTE)]
        axis.plot(frame.epoch, frame.train_loss, linestyle="--", alpha=.65, color=color, label="train")
        axis.plot(frame.epoch, frame.val_loss, marker="o", ms=3, color=color, label="validation")
        best = frame.loc[frame.val_loss.idxmin()]
        axis.scatter(best.epoch, best.val_loss, marker="*", s=130, color="#c19022", edgecolor="white", zorder=4)
        axis.set(title=name, xlabel="эпоха", ylabel="Кросс-энтропия")
        axis.grid(axis="y")
        axis.legend(fontsize=9)
    for axis in list(axes.flat)[count:]:
        axis.axis("off")
    figure.suptitle("Одна шкала для всех моделей  |  звезда = минимальный validation-loss", fontsize=14)
    plt.show()


def plot_comparison(frame):
    plot_metric_panels(frame, columns=["macro_f1", "parameters", "train_s", "batch1_ms"],
                       title="Качество и стоимость одного запуска")


def show_class_metrics(labels, probabilities, title=""):
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(10), zero_division=0)
    frame = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=CLASSES)
    show_table(frame)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10), normalize="true")
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="GnBu")
    for (row, column), value in np.ndenumerate(matrix):
        if value >= .05 or row == column:
            axis.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                      color="white" if value > .55 else "#20364d")
    axis.set_xticks(np.arange(-.5, 10, 1), minor=True)
    axis.set_yticks(np.arange(-.5, 10, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", length=0)
    axis.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="предсказанный класс", ylabel="истинный класс", title=title)
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right")
    figure.colorbar(image, ax=axis, label="доля объектов истинного класса")
    figure.tight_layout()
    plt.show()
    return frame


def corrupt_images(images, kind, severity, seed=123):
    """Фиксированные искажения RGB uint8; нормализацию применяют после них."""
    import cv2
    rng = np.random.default_rng(seed)
    values = images.astype(np.float32) / 255.0
    if kind == "clean":
        changed = values.copy()
    elif kind == "brightness":
        changed = values * (1 - 0.2 * severity)
    elif kind == "noise":
        changed = values + rng.normal(0, 0.04 * severity, size=values.shape)
    elif kind == "shift":
        shift = 2 * severity
        matrix = np.float32([[1, 0, shift], [0, 1, shift]])
        changed = np.stack([cv2.warpAffine(image, matrix, (32, 32),
                           borderMode=cv2.BORDER_REFLECT_101) for image in values])
    elif kind == "blur":
        sigma = 0.5 * severity
        changed = np.stack([cv2.GaussianBlur(image, (5, 5), sigma) for image in values])
    else:
        raise ValueError(kind)
    clipped = np.clip(changed, 0, 1)
    result = np.rint(clipped * 255).astype(np.uint8)
    return result


def paired_accuracy_interval(labels, plain, augmented, seed=123, repeats=1000):
    """Парный bootstrap объектов: условный интервал для двух уже обученных моделей."""
    plain_correct = plain.argmax(axis=1) == labels
    augmented_correct = augmented.argmax(axis=1) == labels
    differences = augmented_correct.astype(float) - plain_correct.astype(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = rng.integers(0, len(labels), size=len(labels))
        estimates.append(differences[sample].mean())
    interval = np.quantile(estimates, [0.025, 0.975])
    result = (float(differences.mean()), float(interval[0]), float(interval[1]))
    return result


def plot_robustness(frame):
    figure, axes = plt.subplots(1, 4, figsize=(17, 4))
    for axis, kind in zip(axes, ["shift", "brightness", "noise", "blur"]):
        selected = frame[frame.corruption == kind]
        for name, group in selected.groupby("model", sort=False):
            axis.plot(group.severity, group.accuracy, marker="o", label=name)
        axis.set(title=kind, xlabel="сила искажения", ylabel="accuracy", ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    plt.show()


def draw_matrices(arrays, titles, annotate=True, cmap="coolwarm", figsize=None, value_limits=None):
    """Одна числовая шкала и одинаковый размер ячейки для всех объектов схемы."""
    from .lesson11_visuals import draw_array_objects
    draw_array_objects(arrays, titles, annotate=annotate, cmap=cmap,
                       figsize=figsize, value_limits=value_limits)


def draw_pipeline(labels, subtitle="", colors=None):
    """Схема последовательных преобразований с подписями форм или операций."""
    if colors is None:
        colors = ["#DCECF9"] * len(labels)
    figure, axis = plt.subplots(figsize=(max(9, 2.1 * len(labels)), 2.9))
    for index, (label, color) in enumerate(zip(labels, colors)):
        axis.text(index, 0, label, ha="center", va="center", fontsize=10,
                  bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "#4C6270"})
        if index < len(labels) - 1:
            axis.annotate("", xy=(index + 0.65, 0), xytext=(index + 0.35, 0),
                          arrowprops={"arrowstyle": "->", "color": "#4C6270"})
    axis.set(xlim=(-0.6, len(labels) - 0.4), ylim=(-0.7, 0.7), title=subtitle)
    axis.axis("off")
    figure.tight_layout()
    plt.show()


def plot_metric_panels(frame, columns=None, title=""):
    """Горизонтальные точки: подписи моделей читаются, единицы не смешиваются."""
    from .lesson11_visuals import PALETTE
    from matplotlib.ticker import PercentFormatter
    if columns is None:
        columns = ["accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "log_loss", "brier"]
    columns = [column for column in columns if column in frame.columns]
    labels = {"accuracy": "Accuracy ↑", "macro_f1": "Macro-F1 ↑", "balanced_accuracy": "Balanced accuracy ↑",
              "top3_accuracy": "Top-3 accuracy ↑", "log_loss": "Log loss ↓", "brier": "Brier ↓",
              "train_s": "Обучение, с ↓", "parameters": "Число параметров ↓", "state_MiB": "Состояние, МиБ ↓",
              "search_wall_s": "Весь поиск, с ↓", "batch1_ms": "Batch=1, мс ↓", "batch128_ms": "Batch=128, мс ↓",
              "images_s": "Картинок/с ↑"}
    rows = (len(columns) + 2) // 3
    height = max(3.2, .36 * len(frame) + 1.1)
    figure, axes = plt.subplots(rows, 3, figsize=(16, height * rows), squeeze=False, layout="constrained")
    positions = np.arange(len(frame))
    colors = [PALETTE[index % len(PALETTE)] for index in positions]
    for axis, column in zip(axes.flat, columns):
        values = frame[column].to_numpy()
        axis.hlines(positions, 0, values, color=colors, alpha=.35, lw=3)
        axis.scatter(values, positions, color=colors, s=55, zorder=3, edgecolors="white")
        axis.set(title=labels.get(column, column), yticks=positions, yticklabels=frame.index.astype(str))
        axis.invert_yaxis()
        axis.set_xlim(0, max(float(values.max()) * 1.24, 1e-6))
        axis.grid(axis="x", alpha=.7)
        proportion = column in ["accuracy", "balanced_accuracy", "top3_accuracy"]
        if proportion:
            axis.xaxis.set_major_formatter(PercentFormatter(1))
        for position, value in zip(positions, values):
            text = f"{value:.1%}" if proportion else f"{value:,.0f}" if column in ["parameters", "images_s"] else f"{value:.3g}"
            axis.annotate(text, (value, position), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    for axis in list(axes.flat)[len(columns):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=15)
    plt.show()
