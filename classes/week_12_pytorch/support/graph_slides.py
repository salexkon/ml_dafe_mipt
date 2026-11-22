"""Пошаговая отрисовка графов из первого занятия по нейросетям."""
import matplotlib.pyplot as plt

BLACK, GREY = "#172D43", "#697C91"
GREEN, RED = "#2E7D32", "#C62828"


def draw_graph(nodes, edges, title=None, size=(13, 5), step=None, ax=None, explanations=None, panel="right"):
    """Вычислительный граф в духе cs231n.

    nodes: {имя: (x, y, подпись, вид)}, вид — "input" (переменная) или "op" (операция);
    edges: [(откуда, куда, значение на прямом проходе, градиент или None)] или с пятым элементом —
    буквенным именем величины на ребре; тогда подписи выглядят как «q = 3» и «∂f/∂q = −4».
    explanations: [(заголовок, текст прямого прохода, текст обратного прохода)] в том же порядке,
    что edges; на текущем шаге пояснение выводится справа от графа (panel="right") или под ним ("below").

    step — сколько шагов показать: None — все; 0 — пустой граф; 1..E — первые step значений
    прямого прохода слева направо; E + 1..2E — все значения и первые step − E градиентов справа
    налево (E — число ребер). Так рисуют анимацию: один вызов на одно состояние.
    """
    order = sorted(range(len(edges)), key=lambda index: nodes[edges[index][1]][0])    # ребра слева направо
    n_edges = len(edges)
    if step is None:
        step = 2 * n_edges
    forward_shown = set(order[:min(step, n_edges)])
    backward_shown = set(order[::-1][:max(0, step - n_edges)])

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=size)
    current = None                                                  # ребро текущего шага
    if 1 <= step <= n_edges:
        current = order[step - 1]
    elif step > n_edges:
        current = order[::-1][step - n_edges - 1]
    for name, (x, y, label, kind) in nodes.items():
        if kind == "op":
            ax.add_patch(plt.Circle((x, y), 0.32, facecolor="white", edgecolor=BLACK, lw=1.5, zorder=3))
            ax.text(x, y, label, ha="center", va="center", fontsize=13, zorder=4)
        else:
            ax.text(x, y, label, ha="right", va="center", fontsize=14, zorder=4)
    for index, edge in enumerate(edges):
        source, target, value, gradient = edge[:4]
        name = edge[4] if len(edge) > 4 else None
        x0, y0 = nodes[source][0], nodes[source][1]
        x1, y1 = nodes[target][0], nodes[target][1]
        start_x = x0 + (0.32 if nodes[source][3] == "op" else 0.1)
        end_x = x1 - (0.32 if nodes[target][3] == "op" else 0.0)
        active_forward = index in forward_shown
        active_backward = index in backward_shown
        is_current = index == current and explanations is not None
        color = GREY if not (active_forward or active_backward) else BLACK
        if is_current:
            color = GREEN if step <= n_edges else RED
        ax.annotate("", xy=(end_x, y1), xytext=(start_x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4 if color == GREY else 2.2 if is_current else 1.8,
                                    shrinkA=0, shrinkB=0))
        fraction = 0.35 if nodes[source][3] == "input" and y0 != y1 else 0.5     # у сходящихся ребер — ближе к началу
        middle_x = start_x + fraction * (end_x - start_x)
        middle_y = y0 + fraction * (y1 - y0)
        slanted = abs(y1 - y0) > 0.3
        offset = 0.26 if slanted else 0.14                                       # у наклонных ребер подписи дальше
        label_size = 14 if name else 12
        if value is not None and active_forward:
            value_text = f"${name} = {value:g}$" if name else f"{value:g}"
            ax.text(middle_x, middle_y + offset, value_text, color=GREEN, ha="center", va="bottom", fontsize=label_size,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
        if gradient is not None and active_backward:
            gradient_text = (r"$\frac{\partial L}{\partial " + name + r"} = " + f"{gradient:g}$") if name else f"{gradient:g}"
            ax.text(middle_x, middle_y - offset, gradient_text, color=RED, ha="center", va="top", fontsize=label_size,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    xs = [x for x, _, _, _ in nodes.values()]
    ys = [y for _, y, _, _ in nodes.values()]
    if explanations is not None:
        panel_x = max(xs) + 0.9
        if current is None:
            text = ("Прямой проход: слева направо считаем\nзначения операций.\n\nОбратный проход: справа налево считаем\n"
                    r"$\frac{\partial L}{\partial\,\cdot}$ по величине на каждом ребре:" "\n"
                    "то, что пришло справа, умножаем\nна локальную производную узла.")
            color = BLACK
        else:
            label, forward_text, backward_text = explanations[current]
            if step <= n_edges:
                text = f"Ребро {label}, прямой проход\n\n{forward_text}"
                color = GREEN
            else:
                text = f"Ребро {label}, обратный проход\n\n{backward_text}"
                color = RED
        box = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=color, lw=1.5)
        if panel == "right":
            ax.text(panel_x, max(ys) + 0.6, text, ha="left", va="top", fontsize=15, color=BLACK, linespacing=2.3, bbox=box)
            ax.set_xlim(min(xs) - 1.2, panel_x + 10.0)
        else:
            ax.text(min(xs) - 0.6, min(ys) - 1.3, text, ha="left", va="top", fontsize=15, color=BLACK, linespacing=2.3, bbox=box)
            ax.set_xlim(min(xs) - 1.2, max(xs) + 1.0)
            ax.set_ylim(min(ys) - 6.2, max(ys) + 0.9)
    else:
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    if step == 0:
        caption = "граф: переменные слева, результат справа"
    elif step <= n_edges:
        caption = f"прямой проход: шаг {step} из {n_edges}, зеленым — значения"
    else:
        caption = f"обратный проход: шаг {step - n_edges} из {n_edges}, красным — градиенты"
    ax.text(min(xs) - 1.1, min(ys) - 0.8, caption, fontsize=10, color=GREY)
    if title:
        ax.set_title(title, fontsize=12)
    if own_figure:
        plt.tight_layout()
        plt.show()


def _figure_png(draw, size):
    """Нарисовать draw(ax) в новую фигуру и вернуть ее PNG; фигура закрывается, inline-бэкенд ее не видит."""
    import io
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    png = buffer.getvalue()
    return png


def _stepper(n_steps, label, render):
    """Слайдер с кнопками; картинка — виджет Image, в который пишут PNG текущего состояния."""
    import ipywidgets as widgets
    from IPython.display import display

    slider = widgets.IntSlider(value=0, min=0, max=n_steps, step=1, description=label, continuous_update=False,
                               layout=widgets.Layout(width="420px"))
    back = widgets.Button(description="< назад", layout=widgets.Layout(width="90px"))
    forward = widgets.Button(description="вперёд >", layout=widgets.Layout(width="90px"))
    image = widgets.Image(format="png", value=render(0))
    back.on_click(lambda _: setattr(slider, "value", max(0, slider.value - 1)))
    forward.on_click(lambda _: setattr(slider, "value", min(n_steps, slider.value + 1)))
    slider.observe(lambda change: setattr(image, "value", render(change["new"])), names="value")
    box = widgets.VBox([widgets.HBox([back, slider, forward]), image])
    display(box)
    return box


def animate_graph(nodes, edges, title=None, size=(13, 5), explanations=None, panel="right"):
    """Виджет: слайдер и кнопки переключают состояние графа из draw_graph; одна картинка на состояние."""
    def render(step):
        png = _figure_png(lambda ax: draw_graph(nodes, edges, title=title, step=step, explanations=explanations,
                                                 panel=panel, ax=ax), size)
        return png

    widget = _stepper(2 * len(edges), "шаг", render)
    return widget
