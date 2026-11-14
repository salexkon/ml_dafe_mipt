"""Наглядные схемы занятия 11; вычисления и обучение остаются в ноутбуках."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from cycler import cycler

INK = "#20364d"
TEAL = "#087f8c"
CORAL = "#da684c"
PURPLE = "#7963ad"
PALETTE = [TEAL, CORAL, PURPLE, "#c19022", "#4085b5", "#405963", "#b45c95", "#788f40", "#87644a", "#5365a5"]


def setup_style():
    """Единая палитра, контрастные подписи и спокойный фон для проектора."""
    plt.rcParams.update({
        "figure.dpi": 120, "figure.facecolor": "white", "axes.facecolor": "#f6f9fc",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlepad": 13,
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#a8b6c4",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=PALETTE), "lines.linewidth": 2,
        "grid.color": "#d7e1eb", "grid.linewidth": .7, "legend.frameon": False,
        "xtick.color": INK, "ytick.color": INK, "savefig.facecolor": "white",
    })


def plot_cifar_pooling(image, maximum, average, window=(12, 10)):
    """RGB до и после pooling: один пиксель занимает одинаковую площадь."""
    from matplotlib.patches import Rectangle

    arrays = [np.asarray(value).transpose(1, 2, 0) for value in [image, maximum, average]]
    titles = ["CIFAR-10  |  вход 32x32", "MaxPool  |  16x16", "AvgPool  |  16x16"]
    widths = [a.shape[1] for a in arrays]
    height = arrays[0].shape[0]
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.set_aspect('equal')
    axis.axis('off')
    left = 0
    row, column = window
    for index, (values, title) in enumerate(zip(arrays, titles)):
        h, width = values.shape[:2]
        top = h / 2
        axis.imshow(values, origin='upper', interpolation='nearest', extent=(left, left + width, -top, top), aspect='equal')
        axis.text(left + width / 2, height / 2 + 3, title, ha='center', weight='bold', fontsize=12)
        axis.text(left + width / 2, -top - 3, '3 канала  |  pooling отдельно по R, G, B' if index == 0 else 'высота и ширина уменьшились вдвое', ha='center', fontsize=9)
        y = top - row - 2 if index == 0 else top - row / 2 - 1
        x = left + column if index == 0 else left + column / 2
        size = 2 if index == 0 else 1
        axis.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor='#efb62e', lw=2))
        left += width + 8
    axis.set(xlim=(-1, left - 7), ylim=(-height / 2 - 6, height / 2 + 7))
    figure.suptitle('Одна и та же картинка  |  одинаковый размер пикселя на трёх изображениях', weight='bold')
    figure.tight_layout()
    plt.show()
    return figure


def draw_batched_linear(maps, flat, weights_transposed, scores):
    """Три карты одного батча переходят в три строки X и Y."""
    from matplotlib.patches import Rectangle

    maps = np.asarray(maps)
    assert maps.shape == (3, 2, 2)
    arrays = [*maps, flat, weights_transposed, scores]
    titles = ["Объект n=0", "Объект n=1", "Объект n=2", "Батч X", r"$W^T$", "Батч Y"]
    figure, axis, boxes = draw_array_objects(arrays, titles, value_limits=(-4, 8),
                                            operators=["", "", "->", "*", "="], show=False,
                                            heading=r"Три объекта: $Y=XW^T$; 3x2x2 -> 3x4 -> 3x3")
    colors = ["#4c72b0", "#da684c", "#7963ad"]
    for sample, color in enumerate(colors):
        box = boxes[sample]
        axis.add_patch(Rectangle((box['left'], box['bottom']), 2, 2, fill=False, ec=color, lw=2, zorder=5))
        for index, columns in [(3, 4), (5, 3)]:
            target = boxes[index]
            top = target['bottom'] + target['height']
            axis.add_patch(Rectangle((target['left'], top - sample - 1), columns, 1,
                                     fill=False, ec=color, lw=1.7, zorder=5))
    left = boxes[0]['left'] - .2
    right = boxes[2]['left'] + 2.2
    axis.plot([left, left, right, right], [-2, -2.25, -2.25, -2], color=INK, lw=1.5)
    axis.text((left + right) / 2, -2.65, "Батч карт: 3x2x2 (N,H,W)", ha='center', va='center', fontsize=11)
    axis.set_ylim(min(axis.get_ylim()[0], -3.2), axis.get_ylim()[1])
    plt.show()
    return figure


def draw_array_objects(arrays, titles, annotate=True, cmap="coolwarm", figsize=None,
                       value_limits=None, highlights=None, heading=None, show=True, operators=None):
    """Матрицы, векторы и скаляры: квадратные ячейки одного размера и общая шкала."""
    import textwrap
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    matrices = []
    for values in arrays:
        matrix = np.asarray(values)
        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Для схемы требуется матрица, вектор или скаляр")
        matrices.append(matrix)
    if len(matrices) != len(titles):
        raise ValueError("Каждому объекту нужна подпись")

    lower = min(float(values.min()) for values in matrices)
    upper = max(float(values.max()) for values in matrices)
    if value_limits is not None:
        lower, upper = value_limits
    if lower == upper:
        lower -= .5
        upper += .5
    if lower < 0 < upper:
        norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    else:
        norm = Normalize(vmin=lower, vmax=upper)
    palette = plt.get_cmap(cmap)

    slots = [max(values.shape[1], 3.2) for values in matrices]
    total_width = sum(slots) + 1.1 * (len(slots) - 1)
    max_rows = max(values.shape[0] for values in matrices)
    logical_height = max_rows + (3.5 if heading else 2.7)
    if figsize is None:
        width_inches = min(19, max(8, (total_width + .8) * .60))
        height_inches = width_inches * logical_height / (total_width + .8)
        figsize = (width_inches, max(2.8, height_inches))
    else:
        figsize = (figsize[0], max(figsize[1], figsize[0] * logical_height / (total_width + .8)))
    figure, axis = plt.subplots(figsize=figsize)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-.4, total_width + .4)
    axis.set_ylim(-max_rows / 2 - .9, max_rows / 2 + (2.6 if heading else 1.8))
    axis.axis("off")
    boxes = []
    cursor = 0.
    for index, (values, title, slot) in enumerate(zip(matrices, titles, slots)):
        rows, columns = values.shape
        left = cursor + (slot - columns) / 2
        bottom = -rows / 2
        top = rows / 2
        axis.imshow(values, cmap=palette, norm=norm, interpolation="nearest", origin="upper",
                    extent=(left, left + columns, bottom, top), aspect="equal", zorder=1)
        if max(rows, columns) <= 24:
            for column in range(columns + 1):
                axis.plot([left + column] * 2, [bottom, top], color="white", lw=.9, alpha=.8, zorder=2)
            for row in range(rows + 1):
                axis.plot([left, left + columns], [bottom + row] * 2, color="white", lw=.9, alpha=.8, zorder=2)
        if annotate and values.size <= 160:
            for (row, column), value in np.ndenumerate(values):
                rgba = palette(norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                axis.text(left + column + .5, top - row - .5, f"{value:.3g}",
                          ha="center", va="center", fontsize=11,
                          color="white" if luminance < .43 else INK, zorder=4)
        for side, direction in [(left - .16, 1), (left + columns + .16, -1)]:
            axis.plot([side + direction * .12, side, side, side + direction * .12],
                      [top + .04, top + .04, bottom - .04, bottom - .04], color=INK, lw=1.25, zorder=3)
        axis.text(left + columns / 2, max_rows / 2 + .65, textwrap.fill(title, width=23),
                  ha="center", va="bottom", fontsize=11, weight="bold", linespacing=1.4)
        axis.text(left + columns / 2, bottom - .38, f"{rows}x{columns}",
                  ha="center", va="top", fontsize=10, color="#6b7d90")
        if highlights and index in highlights:
            for row, column, height, width, color in highlights[index]:
                axis.add_patch(Rectangle((left + column + .025, top - row - height + .025),
                                         width - .05, height - .05, fill=False,
                                         edgecolor=color, linewidth=2.8, zorder=5))
        boxes.append({"left": left, "bottom": bottom, "width": columns, "height": rows})
        cursor += slot + 1.1
    if operators:
        for left_box, right_box, symbol in zip(boxes, boxes[1:], operators):
            middle = (left_box["left"] + left_box["width"] + right_box["left"]) / 2
            if symbol in ["->", "->"]:
                axis.annotate("", (middle + .35, 0), (middle - .35, 0),
                              arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.7})
            else:
                label = r"$\cdot$" if symbol in ["*", "x", "x"] else symbol
                axis.text(middle, 0, label, ha="center", va="center", fontsize=20, color=INK)
    if heading:
        axis.text(total_width / 2, max_rows / 2 + 2., heading, ha="center", va="bottom",
                  fontsize=14, weight="bold")
    figure.subplots_adjust(left=.015, right=.985, bottom=.04, top=.96)
    if show:
        plt.show()
    result = (figure, axis, boxes)
    return result


def plot_bn_tensor(values, normalized):
    """Плоскости HxW сохраняют геометрию; цвет объединяет один канал по N."""
    values = np.asarray(values)
    normalized = np.asarray(normalized)
    batch, channels, height, width = values.shape
    figure = plt.figure(figsize=(14, 6.8), layout="constrained")
    grid = figure.add_gridspec(channels, batch + 1, width_ratios=[1.15] + [1] * batch)
    for channel in range(channels):
        color = ["#4c72b0", CORAL][channel % 2]
        data = values[:, channel]
        summary = figure.add_subplot(grid[channel, 0])
        summary.axis("off")
        summary.text(0, .85, f"Канал C={channel}", color=color, weight="bold", fontsize=14)
        summary.text(0, .55, f"{batch} x {height} x {width} = {data.size} значений\n"
                     f"mu = {data.mean():.2f}\nsigma^2 = {data.var():.2f}", linespacing=1.7)
        summary.text(0, .05, "Одна пара mu, sigma^2\nна ряд одного канала", color=color)
        for sample in range(batch):
            axis = figure.add_subplot(grid[channel, sample + 1])
            axis.imshow(data[sample], cmap="coolwarm", vmin=values.min(), vmax=values.max())
            for (row, column), value in np.ndenumerate(data[sample]):
                axis.text(column, row, f"{value:g}\n-> {normalized[sample, channel, row, column]:+.2f}",
                          ha="center", va="center", fontsize=10,
                          bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 2})
            axis.set(title=f"Объект N={sample}", xlabel=f"W = {width}", ylabel=f"H = {height}")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)
    figure.suptitle("BatchNorm2d  |  NxCxHxW = " + "x".join(map(str, values.shape)) +
                     "\nУсредняем по N, H, W; каналы C не смешиваем. Число -> нормированное число",
                     fontsize=15)
    plt.show()
    return figure


def plot_bn_block(values):
    """Положение BN в исполняемом блоке и агрегация одного канала по батчу."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    batch, channels, height, width = np.asarray(values).shape
    figure, axis = plt.subplots(figsize=(14, 7.2))
    axis.set(xlim=(-.5, 14.5), ylim=(-2.1, 5.7), aspect="equal")
    axis.axis("off")
    labels = ["Вход\n3x2x2x3", "Conv2d 1x1\n2 -> 2 канала", "Выход Conv\n3x2x2x3", "BatchNorm2d(2)\naffine=False\n3x2x2x3", "ReLU\n3x2x2x3"]
    for i, label in enumerate(labels):
        x = i * 2.9
        axis.add_patch(FancyBboxPatch((x, 3.6), 2.3, 1.1, boxstyle="round,pad=.12",
                                     fc="#eaf0f8", ec="#4c72b0", lw=1.5))
        axis.text(x + 1.15, 4.15, label, ha="center", va="center", fontsize=10)
        if i < 4:
            axis.annotate("", (x + 2.78, 4.15), (x + 2.45, 4.15),
                          arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.7})
    axis.text(6.9, 5.25, "NxCxHxW  |  BatchNorm между свёрткой и активацией", ha="center", weight="bold", fontsize=14)
    axis.annotate("", (7.1, 2.9), (7.1, 3.45), arrowprops={"arrowstyle": "->", "color": INK})
    axis.text(7.1, 2.65, "Что поступает в BatchNorm: три объекта, в каждом две карты 2x3", ha="center", fontsize=11)
    for sample in range(batch):
        x = 4.4 + sample * 3.2
        axis.text(x + .9, 2.05, f"Объект n={sample}", ha="center", weight="bold")
        for channel in range(channels):
            y = .85 - channel * 1.8
            color = ["#4c72b0", CORAL][channel % 2]
            for row in range(height):
                for column in range(width):
                    axis.add_patch(Rectangle((x + column * .6, y - row * .6), .6, .6,
                                             fc=color, ec="white", alpha=.17, lw=1))
            axis.add_patch(Rectangle((x, y - .6), 1.8, 1.2, fill=False, ec=color, lw=2))
            axis.text(x + .9, y, f"канал c={channel}", ha="center", va="center", color=color)
    for channel in range(channels):
        data = values[:, channel]
        color = ["#4c72b0", CORAL][channel % 2]
        y = .85 - channel * 1.8
        axis.text(0, y, f"c={channel}: одна пара mu, v\n{batch} | {height} | {width} = {data.size} чисел",
                  va="center", color=color, fontsize=11)
        axis.annotate("", (4.2, y), (3.4, y), arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        axis.plot([4.4, 12.6], [y - .8, y - .8], color=color, lw=2)
    figure.tight_layout()
    plt.show()
    return figure


def plot_group_channels(values, groups):
    """Фиксированное разбиение каналов, независимые группы у каждого объекта."""
    from matplotlib.patches import FancyBboxPatch

    batch, channels, height, width = values.shape
    group_width = channels // groups
    colors = ["#4c72b0", CORAL, "#7963ad", "#c19022"]
    figure, axes = plt.subplots(1, batch, figsize=(13, 5), sharey=True)
    for sample, axis in enumerate(np.atleast_1d(axes)):
        axis.set(xlim=(-.2, groups * 2.35), ylim=(-.7, 4.3))
        axis.axis("off")
        axis.set_title(f"Объект n={sample}  |  группы не выходят за его границы")
        for group in range(groups):
            left = group * 2.35
            color = colors[group % len(colors)]
            axis.add_patch(FancyBboxPatch((left, -.3), 1.85, 3.7, boxstyle="round,pad=.12",
                                         fc=color, alpha=.12, ec=color, lw=2))
            axis.text(left + .925, 3.8, f"Группа {group}", ha="center", color=color, weight="bold")
            group_data = values[sample, group * group_width:(group + 1) * group_width]
            for local_channel in range(group_width):
                channel = group * group_width + local_channel
                axis.text(left + .925, 2.8 - local_channel * .8, f"канал {channel}\n{height}x{width}",
                          ha="center", va="center")
            axis.text(left + .925, .35, f"{group_data.size} чисел\nmu={group_data.mean():.1f}\nv={group_data.var():.2f}",
                      ha="center", va="center", fontsize=10, color=color)
    figure.suptitle(f"GroupNorm  |  C={channels}, G={groups}  |  по {group_width} канала в группе", fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, .93))
    plt.show()
    return figure


def plot_dropout_network(mask_outputs, probability=.5):
    """Каждая маска получена реальным nn.Dropout на единичных активациях."""
    sizes = [4, len(mask_outputs[0][0]), len(mask_outputs[0][1]), 2]
    full = [np.ones(size) for size in sizes]
    masks = [full]
    for output in mask_outputs:
        masks.append([np.ones(sizes[0]), *[np.asarray(item) for item in output], np.ones(sizes[-1])])
    figure, axes = plt.subplots(1, 3, figsize=(14, 6))
    headings = ["eval  |  все нейроны", "train  |  маска A", "train  |  следующий forward"]
    for axis, mask, heading in zip(axes, masks, headings):
        positions = [np.column_stack([np.linspace(-(size - 1) / 2, (size - 1) / 2, size),
                                      np.full(size, layer * 1.8)]) for layer, size in enumerate(sizes)]
        for layer in range(len(sizes) - 1):
            for left, start in enumerate(positions[layer]):
                for right, end in enumerate(positions[layer + 1]):
                    if mask[layer][left] and mask[layer + 1][right]:
                        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                                shrinkA=11, shrinkB=11, color=INK, alpha=.3, lw=.9)
                        axis.add_patch(arrow)
        for layer, points in enumerate(positions):
            for index, point in enumerate(points):
                active = mask[layer][index] > 0
                color = TEAL if active else CORAL
                axis.add_patch(Circle(point, .19, facecolor="white", edgecolor=color, lw=2, zorder=3))
                if not active:
                    axis.plot(*point, marker="x", color=CORAL, ms=9, mew=1.8, zorder=4)
        axis.set(xlim=(-3.5, 3.5), ylim=(-.6, 6), title=heading)
        axis.axis("off")
        note = "h -> h" if heading.startswith("eval") else f"сохранённые h -> h / (1-p) = {1/(1-probability):g}h"
        axis.text(.5, -.02, note, transform=axis.transAxes, ha="center", fontsize=11, color=TEAL)
    figure.suptitle(f"Dropout p={probability:g}  |  выключаем скрытые активации и их связи на один forward",
                     fontsize=14, weight="bold")
    figure.text(.5, .02, "Веса остаются в модели. Новая маска не создаёт новую обучаемую сеть.", ha="center")
    figure.tight_layout(rect=(0, .08, 1, .94))
    plt.show()
    return figure


def plot_feature_atlas(image, feature_maps):
    """Фактические активации выбранных слоёв, единый масштаб внутри слоя."""
    figure, axes = plt.subplots(len(feature_maps), 7, figsize=(14, 3 * len(feature_maps)),
                                layout="constrained", squeeze=False)
    cmap = LinearSegmentedColormap.from_list("activation", ["#101f35", "#177d8b", "#ffe8aa"])
    for row, (name, values) in enumerate(feature_maps.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Вход RGB", fontsize=10)
        maximum = max(float(np.max(values)), 1e-8)
        for channel in range(6):
            axis = axes[row, channel + 1]
            axis.imshow(values[channel], cmap=cmap, vmin=0, vmax=maximum, interpolation="nearest")
            zeros = (values[channel] == 0).mean()
            axis.set_title(f"c={channel}  |  нули {zeros:.0%}", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(name + f"\n{values.shape[1]}x{values.shape[2]}", fontsize=11)
    figure.suptitle("Что передаётся между слоями обученной CNN\nТёмное = 0; светлое = сильный отклик. Масштаб общий внутри строки.", fontsize=15)
    plt.show()
    return figure


def plot_pooling(values, maximum, average, max_gradient, mean_gradient):
    """Окна, два вида pooling и градиенты на общей числовой шкале."""
    arrays = [values, maximum, average, max_gradient, mean_gradient]
    titles = ["Вход: окна 2x2", "MaxPool  |  максимум", "AvgPool  |  среднее",
              "dsum(MaxPool) / dX", "dsum(AvgPool) / dX"]
    windows = [(row, column, 2, 2, PALETTE[index])
               for index, (row, column) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)])]
    figure, axis, boxes = draw_array_objects(
        arrays, titles, cmap="GnBu", highlights={0: windows},
        heading="Pooling: значения и градиенты", show=False)
    plt.show()
    return figure


def export_tradeoff(frame, output):
    """Автономный Plotly: качество-стоимость, размер точки - параметры."""
    import plotly.graph_objects as go
    names = frame.index.astype(str).tolist()
    metrics = [("train_s", "Время обучения, с"), ("batch1_ms", "Forward batch=1, мс"),
               ("parameters", "Обучаемые параметры")]
    custom = np.column_stack([frame.accuracy, frame.macro_f1, frame.parameters,
                              frame.train_s, frame.batch1_ms])
    sizes = 12 + 22 * np.sqrt(frame.parameters / frame.parameters.max())
    figure = go.Figure(go.Scatter(
        x=frame.train_s, y=frame.accuracy, mode="markers", text=names, customdata=custom,
        marker={"size": sizes, "color": (PALETTE * ((len(names) + len(PALETTE) - 1) // len(PALETTE)))[:len(names)],
                "line": {"color": "white", "width": 1.5}, "opacity": .9},
        hovertemplate="<b>%{text}</b><br>Accuracy: %{customdata[0]:.1%}<br>Macro-F1: %{customdata[1]:.3f}"
                      "<br>Параметры: %{customdata[2]:,.0f}<br>Обучение: %{customdata[3]:.2f} с"
                      "<br>Batch=1: %{customdata[4]:.3f} мс<extra></extra>"))
    buttons = []
    for column, title in metrics:
        buttons.append({"label": title, "method": "update", "args": [
            {"x": [frame[column].tolist()]}, {"xaxis.title.text": title}]})
    figure.update_layout(template="plotly_white", height=580, font={"family": "Arial", "color": INK},
                         title="Качество и стоимость  |  наведите на модель, выберите горизонтальную ось",
                         xaxis={"title": metrics[0][1], "rangemode": "tozero"},
                         yaxis={"title": "Test accuracy  |  выше лучше", "tickformat": ".0%", "range": [0, 1]},
                         updatemenus=[{"buttons": buttons, "direction": "right", "x": 0, "y": 1.16}],
                         margin={"t": 130, "b": 85}, annotations=[{
                             "text": "Предпочтительно выше и левее  |  площадь точки отражает число параметров приблизительно",
                             "x": 0, "y": - .17, "xref": "paper", "yref": "paper", "showarrow": False}])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True)
    return figure


def plot_tradeoff_preview(frame):
    """Статическая карта с легендой вне поля: близкие точки не закрыты подписями."""
    from matplotlib.ticker import PercentFormatter
    from matplotlib.lines import Line2D
    figure, axis = plt.subplots(figsize=(13, 5.5), layout="constrained")
    handles = []
    for index, (name, row) in enumerate(frame.iterrows()):
        color = PALETTE[index % len(PALETTE)]
        marker = "o" if index < len(PALETTE) else "D"
        size = 65 + 300 * np.sqrt(row.parameters / frame.parameters.max())
        axis.scatter(row.train_s, row.accuracy, s=size, color=color, marker=marker,
                     alpha=.85, edgecolor="white", zorder=3)
        handles.append(Line2D([], [], marker=marker, linestyle="", markersize=7,
                              color=color, label=str(name)))
    axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=9,
                title="Модель  |  точные значения по наведению в HTML", title_fontsize=9)
    axis.set(xlabel="Время обучения, с  |  меньше лучше", ylabel="Test accuracy  |  больше лучше",
             title="Качество при ограниченном времени  |  размер точки отражает число параметров",
             xlim=(0, frame.train_s.max() * 1.12), ylim=(max(0, frame.accuracy.min() - .10),
                                                        min(1, frame.accuracy.max() + .10)))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=.65)
    plt.show()
    return figure


def plot_corruption_facets(frame, metric="accuracy"):
    """Разделяем четыре типа искажений; общий severity не соединяет разные условия."""
    titles = {"shift": "Сдвиг", "brightness": "Затемнение", "noise": "Шум", "blur": "Размытие"}
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True, layout="constrained")
    for axis, (kind, title) in zip(axes, titles.items()):
        for index, (name, group) in enumerate(frame.groupby("model", sort=False)):
            group = group.copy()
            group["kind"] = group.condition.str.rsplit("_", n=1).str[0]
            selected = group[group.kind == kind].copy()
            selected["level"] = selected.condition.str.rsplit("_", n=1).str[-1].astype(int)
            selected = selected.sort_values("level")
            clean = float(group.loc[group.condition == "clean", metric].iloc[0])
            xs = np.r_[0, selected.level]
            ys = np.r_[clean, selected[metric]]
            axis.plot(xs, ys, marker=["o", "s"][index % 2], label=name, color=PALETTE[index])
        axis.set(title=title, xlabel="Сила искажения", xticks=[0, 1, 2, 3])
        axis.grid(axis="y")
    if metric != "log_loss":
        axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(metric + (" ↓" if metric == "log_loss" else " ↑"))
    axes[-1].legend(fontsize=9)
    figure.suptitle("Устойчивость  |  0 = чистый test; в каждой панели меняется только одно искажение", fontsize=13)
    plt.show()
    return figure


def export_matrix_windows(inputs, kernel, output):
    """Слайдер связывает окно X, строку A(K) и один выход Y."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    height, width = inputs.shape
    kh, kw = kernel.shape
    oh, ow = height - kh + 1, width - kw + 1
    result = np.empty((oh, ow))
    operator = np.zeros((oh * ow, height * width))
    for position in range(oh * ow):
        row, column = divmod(position, ow)
        result[row, column] = (inputs[row:row + kh, column:column + kw] * kernel).sum()
        for u in range(kh):
            for v in range(kw):
                operator[position, (row + u) * width + column + v] = kernel[u, v]
    figure = make_subplots(rows=1, cols=3, column_widths=[.24, .52, .24],
                           subplot_titles=["Вход X и выбранное окно", "A(K)  |  одна строка = одно окно", "Выход Y"])
    for column, values in enumerate([inputs, operator, result], 1):
        figure.add_trace(go.Heatmap(z=values, text=values, texttemplate="%{text:g}",
                                   colorscale="Blues", showscale=False,
                                   hovertemplate="строка %{y}, столбец %{x}: %{z}<extra></extra>"), row=1, col=column)
    steps = []
    for position in range(oh * ow):
        row, column = divmod(position, ow)
        shapes = [
            {"type": "rect", "xref": "x", "yref": "y", "x0": column - .5, "x1": column + kw - .5,
             "y0": row - .5, "y1": row + kh - .5, "line": {"color": CORAL, "width": 4}},
            {"type": "rect", "xref": "x2", "yref": "y2", "x0": -.5, "x1": height * width - .5,
             "y0": position - .5, "y1": position + .5, "line": {"color": CORAL, "width": 3}},
            {"type": "rect", "xref": "x3", "yref": "y3", "x0": column - .5, "x1": column + .5,
             "y0": row - .5, "y1": row + .5, "line": {"color": CORAL, "width": 4}}]
        products = inputs[row:row + kh, column:column + kw] * kernel
        expression = " + ".join(f"{value:g}" for value in products.flat)
        steps.append({"label": str(position), "method": "relayout", "args": [{
            "shapes": shapes, "title.text": f"Окно {position}: Y[{row},{column}] = {expression} = {result[row,column]:g}"}]})
    figure.update_layout(template="plotly_white", height=590, font={"family": "Arial", "color": INK},
                         title=steps[0]["args"][0]["title.text"], shapes=steps[0]["args"][0]["shapes"],
                         sliders=[{"steps": steps, "currentvalue": {"prefix": "Номер окна: "}, "pad": {"t": 45}}])
    figure.update_yaxes(autorange="reversed", showticklabels=False)
    figure.update_xaxes(showticklabels=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True)
    return figure


def show_convolution_window(X, K, Y, position=0):
    row, column = divmod(position, Y.shape[1])
    kernel_height, kernel_width = K.shape
    window = X[row:row + kernel_height, column:column + kernel_width]
    products = window * K
    arrays = [X, window, K, products, Y]
    titles = ["Вход X", "Окно X", "Ядро K", "Поэлементные произведения", "Выход Y"]
    # Диапазон произведений учитывает все положения окна, а не только выбранное.
    extreme_products = [float(x) * float(k) for x in [X.min(), X.max()] for k in [K.min(), K.max()]]
    lower = min(float(X.min()), float(K.min()), float(Y.min()), min(extreme_products))
    upper = max(float(X.max()), float(K.max()), float(Y.max()), max(extreme_products))
    highlights = {
        0: [(row, column, kernel_height, kernel_width, "#c19022")],
        4: [(row, column, 1, 1, "#c19022")],
    }
    draw_array_objects(arrays, titles, value_limits=(lower, upper), highlights=highlights,
                       heading=f"Y[{row}, {column}] = сумма произведений = {products.sum():g}")


def plot_adaptive_regions(inputs, pooled, selected=(1, 0)):
    row, column = selected
    oh, ow = pooled.shape
    height, width = inputs.shape
    start_h = row * height // oh
    end_h = ((row + 1) * height + oh - 1) // oh
    start_w = column * width // ow
    end_w = ((column + 1) * width + ow - 1) // ow
    highlights = {0: [(start_h, start_w, end_h - start_h, end_w - start_w, TEAL)],
                  2: [(row, column, 1, 1, TEAL)]}
    draw_array_objects([inputs, inputs[start_h:end_h, start_w:end_w], pooled],
                       ["Вход X", "Выбранная область", "AdaptiveAvgPool: выход Y"],
                       highlights=highlights,
                       heading=f"Y[{row},{column}] = mean(X[{start_h}:{end_h}, {start_w}:{end_w}]) = {pooled[row,column]:g}")


def plot_pool_timings(frame):
    cases = frame["вход"].drop_duplicates().tolist()
    devices = frame["устройство"].drop_duplicates().tolist()
    figure, axes = plt.subplots(len(devices), len(cases), figsize=(15, 3.5 * len(devices)),
                                squeeze=False, layout="constrained")
    for row, device in enumerate(devices):
        for column, case in enumerate(cases):
            axis = axes[row, column]
            selected = frame[(frame["вход"] == case) & (frame["устройство"] == device)]
            positions = np.arange(len(selected))
            values = selected["медиана, мкс"].to_numpy()
            axis.hlines(positions, 0, values, color=PALETTE[:len(selected)], lw=3, alpha=.35)
            axis.scatter(values, positions, color=PALETTE[:len(selected)], s=55)
            for position, value in zip(positions, values):
                axis.annotate(f"{value:.1f}", (value, position), xytext=(6, 0), textcoords="offset points", va="center")
            axis.set(yticks=positions, yticklabels=selected["операция"], xlabel="Forward, мкс",
                     title=f"{device}  |  {case} -> 4x4", xlim=(0, max(values.max() * 1.3, 1)))
            axis.invert_yaxis()
            axis.grid(axis="x")
    figure.suptitle("Одинаковые вход и размер выхода  |  прогрев и синхронизация устройства", fontsize=14)
    plt.show()
    return figure


def plot_receptive_jump(receptive_field=18, jump=4):
    from matplotlib.patches import Rectangle
    first_center = (receptive_field - 1) / 2
    second_center = first_center + jump
    figure, axis = plt.subplots(figsize=(12, 3.7), layout="constrained")
    for level, start, color, label in [(1, 0, TEAL, "Выход u"), (0, jump, PURPLE, "Выход u+1")]:
        axis.add_patch(Rectangle((start - .5, level - .24), receptive_field, .48,
                                 facecolor=color, edgecolor=color, alpha=.22))
        center = start + first_center
        axis.scatter(center, level, color=color, s=60, zorder=3)
        axis.annotate(f"r={receptive_field}", (center, level + .26), ha="center", color=color)
        axis.text(-1, level, label, ha="right", va="center")
    axis.vlines([first_center, second_center], 0, 1.65, color=[TEAL, PURPLE], ls="--", alpha=.6)
    axis.annotate("", (second_center, 1.5), (first_center, 1.5),
                  arrowprops={"arrowstyle":"<->", "color":INK, "lw":1.5})
    axis.text((first_center + second_center)/2, 1.65, f"j={jump} пикселя исходного входа", ha="center")
    axis.set(xlim=(-4, receptive_field + jump), ylim=(-.5, 2), yticks=[],
             xticks=np.arange(0, receptive_field + jump, 2), xlabel="Позиция в исходном изображении",
             title="Два соседних элемента выходной карты  |  схема без padding")
    axis.spines["left"].set_visible(False)
    plt.show()
    return figure


def plot_size_cost(frame):
    from matplotlib.ticker import FuncFormatter
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    settings = [("CNN", "#4c72b0", "o", "-"), ("MLP", CORAL, "D", "--")]
    for name, color, marker, style in settings:
        axes[0].plot(frame["сторона S"], frame[name + " параметры"], color=color, marker=marker,
                     linestyle=style, label=name)
        axes[1].plot(frame["сторона S"], frame[name + " MAC"], color=color, marker=marker,
                     linestyle=style, label=name)
    axes[0].set(yscale="log", ylabel="Обучаемые параметры  |  log", title="Параметры модели для входа SxS")
    axes[1].set(ylabel="MAC на одно изображение", title="Вычисления Conv и Linear")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g} млн"))
    for axis in axes:
        axis.set(xlabel="Сторона изображения S", xticks=[8, 32, 64, 96, 128])
        axis.grid(axis="y")
        axis.legend()
    axes[0].annotate("618  |  те же веса", (128, 618), xytext=(-8, 12), textcoords="offset points",
                     ha="right", color="#4c72b0")
    plt.show()
    return figure


def plot_resolution_examples(variants):
    import textwrap
    max_height = max(image.shape[0] for image in variants.values())
    total_width = sum(image.shape[1] for image in variants.values()) + 8 * (len(variants) - 1)
    figure, axis = plt.subplots(figsize=(13, 4.4))
    axis.set(xlim=(-2, total_width + 2), ylim=(-max_height/2-7, max_height/2+16))
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    left = 0
    for name, image in variants.items():
        height, width = image.shape[:2]
        axis.imshow(image, extent=(left, left+width, -height/2, height/2), interpolation="nearest")
        axis.text(left+width/2, max_height/2+7, textwrap.fill(name, 20),
                  ha="center", va="bottom", weight="bold", fontsize=11)
        axis.text(left+width/2, -height/2-3, f"{height}x{width}", ha="center", va="top", color="#6b7d90")
        left += width + 8
    figure.tight_layout()
    plt.show()
    return figure


def plot_bn_batch_effect(batch_a, batch_b, train_outputs, eval_outputs, *,
                         gamma, beta, running_mean, running_variance, momentum=.25, eps=1e-5):
    """Полные входы/выходы и все подстановки для первого объекта, канала 0."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    batches = [np.asarray(batch_a), np.asarray(batch_b)]
    train_outputs, eval_outputs = np.asarray(train_outputs), np.asarray(eval_outputs)
    gamma, beta = np.asarray(gamma), np.asarray(beta)
    running_mean, running_variance = np.asarray(running_mean), np.asarray(running_variance)
    assert batches[0].shape == batches[1].shape == (4, 2)
    assert train_outputs.shape == eval_outputs.shape == (2, 4, 2)
    for index, values in enumerate(batches):
        expected = gamma * (values - values.mean(axis=0)) / np.sqrt(values.var(axis=0) + eps) + beta
        assert np.allclose(train_outputs[index], expected, atol=1e-6)
        expected_eval = gamma * (values - running_mean) / np.sqrt(running_variance + eps) + beta
        assert np.allclose(eval_outputs[index], expected_eval, atol=1e-6)
    mean_a = batches[0].mean(axis=0)
    unbiased_a = batches[0].var(axis=0, ddof=1)
    assert np.allclose(running_mean, momentum * mean_a)
    assert np.allclose(running_variance, (1 - momentum) + momentum * unbiased_a)

    all_values = np.concatenate([a.ravel() for a in batches] + [train_outputs.ravel(), eval_outputs.ravel()])
    norm = TwoSlopeNorm(vmin=min(float(all_values.min()), -.1), vcenter=0, vmax=float(all_values.max()))
    palette = plt.get_cmap('coolwarm')
    gold = '#c19022'
    figure = plt.figure(figsize=(19, 13.5), layout='constrained')
    grid = figure.add_gridspec(3, 2, height_ratios=[1.25, .9, 1.65], hspace=.12, wspace=.08)

    def matrix(axis, values, left, title):
        axis.imshow(values, cmap=palette, norm=norm, origin='upper', interpolation='nearest',
                    extent=(left, left + 2, 0, 4), aspect='equal')
        for row in range(5):
            axis.plot([left, left + 2], [row, row], color='white', lw=1)
        axis.plot([left + 1, left + 1], [0, 4], color='white', lw=1)
        for (row, col), value in np.ndenumerate(values):
            rgba = palette(norm(value))
            brightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            axis.text(left + col + .5, 3.5 - row, f'{value:.3g}', ha='center', va='center',
                      fontsize=12, color='white' if brightness < .43 else INK)
        axis.add_patch(Rectangle((left, 3), 1, 1, fill=False, ec=gold, lw=3))
        axis.add_patch(Rectangle((left, 0), 2, 4, fill=False, ec='#9eafc0', lw=1))
        axis.text(left + 1, 5.0, title, ha='center', weight='bold', fontsize=13)
        axis.text(left + .5, 4.25, 'c=0', ha='center', fontsize=11)
        axis.text(left + 1.5, 4.25, 'c=1', ha='center', fontsize=11)
        axis.text(left + 1, -.45, '4x2: объекты x каналы', ha='center', fontsize=10, color='#61778c')
        for row in range(4):
            axis.text(left - .18, 3.5 - row, f'n={row}', ha='right', va='center', fontsize=10)

    for index, name in enumerate(['A', 'B']):
        axis = figure.add_subplot(grid[0, index])
        axis.set(xlim=(-.8, 6.8), ylim=(-.9, 5.7), aspect='equal')
        axis.axis('off')
        matrix(axis, batches[index], 0, f'Вход X_{name}')
        matrix(axis, train_outputs[index], 4.2, f'Выход train Y_{name}')
        axis.annotate('', (3.7, 2), (2.35, 2), arrowprops={'arrowstyle': '->', 'lw': 1.7, 'color': INK})

        calc = figure.add_subplot(grid[1, index])
        calc.set(xlim=(0, 1), ylim=(0, 1)); calc.axis('off')
        values = batches[index][:, 0]
        mean, variance = values.mean(), values.var()
        deviations = values - mean
        mean_sum = '+'.join(f'{v:g}' for v in values)
        squares = '+'.join(f'{v*v:g}' for v in deviations)
        calc.text(.02, .96, f'Расчёт для c=0, батч {name}', weight='bold', fontsize=15, va='top')
        lines = [rf'$\mu_{name}=({mean_sum})/4={mean:g}$',
                 rf'$v_{name}=[\sum_i(x_i-\mu_{name})^2]/4=({squares})/4={variance:g}$',
                 rf'$y^{{train}}_{name}[0,0]={gamma[0]:g}\cdot\frac{{{values[0]:g}-{mean:g}}}{{\sqrt{{{variance:g}+\varepsilon}}}}+{beta[0]:g}={train_outputs[index,0,0]:.4f}$']
        for y, line in zip([.69, .43, .14], lines):
            calc.text(.02, y, line, fontsize=15, va='center')

    running = figure.add_subplot(grid[2, 0])
    running.set(xlim=(0, 1), ylim=(0, 1)); running.axis('off')
    running.text(.02, .98, 'Откуда взялись сохранённые статистики', fontsize=15, weight='bold', va='top')
    running.text(.02, .85, 'Исходный слой: один train-проход батча A, momentum=0.25.\nДо него running_mean=[0,0], running_var=[1,1].', fontsize=12, linespacing=1.5, va='top')
    running_lines = [r'$\mu_{run,0}=0.75\cdot0+0.25\cdot4=1$',
                     r'$s^2_{A,0}=(9+1+1+9)/(4-1)=20/3$',
                     r'$v_{run,0}=0.75\cdot1+0.25\cdot(20/3)=29/12\approx2.4167$',
                     r'$\mu_{A,1}=(10+14+18+22)/4=16$',
                     r'$s^2_{A,1}=(36+4+4+36)/(4-1)=80/3$',
                     r'$\mu_{run,1}=0.75\cdot0+0.25\cdot16=4$',
                     r'$v_{run,1}=0.75\cdot1+0.25\cdot(80/3)=89/12\approx7.4167$']
    for y, line in zip(np.linspace(.64, .06, len(running_lines)), running_lines):
        running.text(.02, y, line, fontsize=14, va='center')

    evaluation = figure.add_subplot(grid[2, 1])
    evaluation.set(xlim=(-.8, 6.8), ylim=(-3.0, 5.7), aspect='equal'); evaluation.axis('off')
    matrix(evaluation, eval_outputs[0], 0, 'Выход eval\nбатч A')
    matrix(evaluation, eval_outputs[1], 4.2, 'Выход eval\nбатч B')
    evaluation.text(3, -1.25, r'$y^{eval}_A[0,0]=2\cdot\frac{1-1}{\sqrt{29/12+\varepsilon}}+0.5=0.5$', ha='center', fontsize=14)
    evaluation.text(3, -2.15, r'$y^{eval}_B[0,0]=2\cdot\frac{1-1}{\sqrt{29/12+\varepsilon}}+0.5=0.5$', ha='center', fontsize=14)
    figure.suptitle('BatchNorm: одинаковые формы входа и выхода, 4x2\n'
                   'Строки - объекты, столбцы - каналы. Золотая рамка: элемент [0,0].\n'
                   'train A и B: независимые копии; eval: исходный слой после одного батча A.\n'
                   'gamma=[2, 0.5], beta=[0.5, -1], eps=0.00001', fontsize=15, weight='bold')
    plt.show()
    return figure
