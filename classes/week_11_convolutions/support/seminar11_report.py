"""Оформление результатов семинара 11; модели и обучение находятся в ноутбуке."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def comparison_style(frame, device):
    columns = [
        "accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "log_loss", "brier",
        "parameters", "train_s", "batch1_ms", "images_s",
    ]
    table = frame[columns].round(10).copy()
    quality = columns[:6]
    cost = columns[6:]
    labels = [
        "Accuracy ↑", "Macro-F1 ↑", "Balanced acc. ↑", "Top-3 ↑", "Log loss ↓", "Brier ↓",
        "Параметры ↓", "Обучение, с ↓", "Batch=1, мс ↓", "Картинок/с ↑",
    ]
    groups = ["Качество на test"] * len(quality) + ["Стоимость"] * len(cost)
    table.columns = pd.MultiIndex.from_arrays([groups, labels])
    table.index.name = "Модель"

    formats = {column: "{:.3f}" for column in table.columns}
    for column in table.columns[:4]:
        formats[column] = "{:.1%}"
    formats[table.columns[6]] = "{:,.0f}"
    formats[table.columns[7]] = "{:.1f}"
    formats[table.columns[9]] = "{:,.0f}"

    style = table.style.format(formats)
    style = style.set_table_attributes('style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px"')
    higher = list(table.columns[:4]) + [table.columns[9]]
    lower = list(table.columns[4:9])
    style = style.background_gradient(cmap="YlGn", subset=higher)
    style = style.background_gradient(cmap="YlGn_r", subset=lower)
    style = style.set_caption(f"CIFAR-10  |  одинаковые данные и 8 эпох  |  устройство: {device}")
    style = style.set_table_styles([
        {"selector": "caption", "props": [("text-align", "left"), ("font-size", "17px"),
                                           ("font-weight", "600"), ("padding", "12px 0")]},
        {"selector": "th.col_heading", "props": [("background-color", "#16324f"),
                                                 ("color", "white"), ("padding", "10px")]},
        {"selector": "th.row_heading", "props": [("text-align", "left"),
                                                 ("background-color", "#edf3f8"),
                                                 ("padding", "9px 12px")]},
        {"selector": "td", "props": [("padding", "9px 10px"), ("text-align", "right"),
                                      ("font-variant-numeric", "tabular-nums"),
                                      ("border-bottom", "1px solid #e2e8f0")]},
        {"selector": "table", "props": [("border-collapse", "collapse"),
                                         ("font-family", "sans-serif"), ("font-size", "13px")]},
    ])
    return style


def show_comparison(frame, device):
    style = comparison_style(frame, device)
    display(style)
    return style


def plot_training_groups(histories, groups):
    for title, names in groups.items():
        figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for index, name in enumerate(names):
            history = pd.DataFrame(histories[name])
            from .lesson11_visuals import PALETTE
            model_order = list(histories)
            color = PALETTE[model_order.index(name) % len(PALETTE)]
            axes[0].plot(history.epoch, history.train_loss, label=name, color=color, marker="o", ms=3)
            axes[1].plot(history.epoch, history.val_loss, label=name, color=color, marker="o", ms=3)
            best = history.loc[history.val_loss.idxmin()]
            axes[1].scatter(best.epoch, best.val_loss, marker="*", color=color, s=95,
                            edgecolor="white", zorder=4)
            axes[2].plot(history.epoch, history.val_accuracy, label=name, color=color, marker="o", ms=3)

        for axis, label in zip(axes, ["Train loss", "Validation loss", "Validation accuracy"]):
            axis.set(xlabel="эпоха", ylabel=label)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        axes[2].set_ylim(0, 1)
        figure.suptitle(title + "  |  звезда = выбранная эпоха", fontsize=13)
        figure.tight_layout()
        plt.show()


def plot_comparison(frame):
    positions = np.arange(len(frame))
    colors = ["#e49b44" if name == "CNN base" else "#417a9b" for name in frame.index]
    figure, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
    panels = [("accuracy", "Test accuracy ↑"), ("parameters", "Параметры ↓, log"),
              ("train_s", "Время обучения, с ↓")]
    for axis, (column, title) in zip(axes, panels):
        values = frame[column].to_numpy()
        axis.barh(positions, values, color=colors, height=.65)
        axis.set_title(title)
        axis.set_yticks(positions, frame.index)
        axis.grid(axis="x", alpha=.2)
        if column == "parameters":
            axis.set_xscale("log")
            axis.set_xlim(max(1, values.min() / 3), values.max() * 6)
        else:
            axis.set_xlim(0, values.max() * 1.22)
        for position, value in zip(positions, values):
            label = f"{value:.1%}" if column == "accuracy" else f"{value:,.0f}" if column == "parameters" else f"{value:.1f}"
            axis.annotate(label, (value, position), xytext=(5, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=9)
    axes[0].invert_yaxis()
    figure.tight_layout()
    plt.show()


def plot_robustness(frame):
    from .lesson11_visuals import TEAL, CORAL, INK
    from matplotlib.ticker import PercentFormatter
    labels = {"clean": "Чистые", "shift": "Сдвиг 4 px", "brightness": "Яркость x0.6",
              "noise": "Шум sigma=0.08", "blur": "Размытие sigma=1"}
    positions = np.arange(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    plain = frame["CNN base"].to_numpy()
    augmented = frame["CNN + aug"].to_numpy()
    axes[0].hlines(positions, np.minimum(plain, augmented), np.maximum(plain, augmented), color="#a8b6c4", lw=3)
    axes[0].scatter(plain, positions, color=TEAL, s=70, label="CNN base", zorder=3)
    axes[0].scatter(augmented, positions, color=CORAL, s=65, marker="D", label="CNN + aug", zorder=3)
    axes[0].set(yticks=positions, yticklabels=[labels[k] for k in frame.index], xlabel="Accuracy ↑",
                xlim=(0, 1), title="Две модели  |  те же изображения")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="lower left")
    delta = 100 * (augmented - plain)
    colors = [TEAL if value >= 0 else CORAL for value in delta]
    axes[1].barh(positions, delta, color=colors, height=.55, alpha=.85)
    axes[1].axvline(0, color=INK, lw=1)
    limit = max(abs(delta).max() * 1.5, 1)
    axes[1].set(xlim=(-limit, limit), xlabel="Разность accuracy, п.п.", title="Аугментации: выигрыш или потеря")
    for position, value in zip(positions, delta):
        axes[1].annotate(f"{value:+.2f}", (value, position), xytext=(5 if value >= 0 else -5, 0),
                         textcoords="offset points", va="center", ha="left" if value >= 0 else "right")
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", alpha=.6)
    plt.show()
