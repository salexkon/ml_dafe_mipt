"""3D-поверхности и шаги оптимизаторов; вычисление траекторий раскрыто в лекции."""
from pathlib import Path
import html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

METHOD_STYLE = {
    "SGD": {"color": "#2463a6", "marker": "o", "symbol": "circle"},
    "momentum": {"color": "#d67a26", "marker": "D", "symbol": "diamond"},
    "Adam": {"color": "#258c69", "marker": "s", "symbol": "square"},
}


def surface_grid(example, count=85):
    (left, right), (bottom, top) = example["bounds"]
    horizontal = np.linspace(left, right, count)
    vertical = np.linspace(bottom, top, count)
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    points = np.stack([grid_x, grid_y], axis=-1)
    losses = example["objective"](points)
    return grid_x, grid_y, losses


def display_height(values, example):
    result = np.log1p(values) if example.get("log_height", False) else values
    return result


def plot_landscape(example):
    grid_x, grid_y, losses = surface_grid(example)
    height = display_height(losses, example)
    max_step = max(len(trace["losses"]) - 1 for trace in example["traces"].values())
    time_norm = Normalize(0, max_step)
    cmap = plt.get_cmap("viridis")

    figure = plt.figure(figsize=(15, 6), layout="constrained")
    axis3d = figure.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    axis2d = figure.add_subplot(1, 2, 2)
    axis3d.plot_surface(grid_x, grid_y, height, color="#cbd3dc", alpha=.13,
                        linewidth=0, antialiased=True, zorder=0)
    axis3d.plot_wireframe(grid_x, grid_y, height, rstride=7, cstride=7,
                          color="#aeb9c7", linewidth=.55, alpha=.4, zorder=1)
    contours = axis2d.contour(grid_x, grid_y, losses, levels=example["levels"],
                              colors="#c2c8d0", linewidths=.8, zorder=0)
    axis2d.clabel(contours, levels=example["label_levels"], fmt="%g", fontsize=8, colors="#737e8a")

    handles = []
    for name, trace in example["traces"].items():
        style = METHOD_STYLE[name]
        points = trace["positions"]
        values = display_height(trace["losses"], example)
        step_numbers = np.arange(len(points))
        axis3d.plot(points[:, 0], points[:, 1], values, color=style["color"],
                     linewidth=1.4, zorder=5)
        axis3d.scatter(points[:, 0], points[:, 1], values, c=step_numbers, cmap=cmap,
                        norm=time_norm, marker=style["marker"], s=17, edgecolors="white",
                        linewidths=.25, depthshade=False, zorder=6)
        axis2d.plot(points[:, 0], points[:, 1], color=style["color"], linewidth=1.25, zorder=2)
        axis2d.scatter(points[:, 0], points[:, 1], c=step_numbers, cmap=cmap, norm=time_norm,
                        marker=style["marker"], s=21, edgecolors="white", linewidths=.25, zorder=3)
        axis3d.scatter(*points[-1], values[-1], color=style["color"], marker="X", s=75, zorder=8)
        axis2d.scatter(*points[-1], color=style["color"], marker="X", s=80, zorder=5)
        handles.append(Line2D([], [], color=style["color"], marker=style["marker"], label=name))

    minimum = np.asarray(example["minimum"])
    minimum_loss = float(example["objective"](minimum))
    axis3d.scatter(*minimum, display_height(minimum_loss, example), color="black", marker="*", s=120, zorder=9)
    axis2d.scatter(*minimum, color="black", marker="*", s=120, zorder=6)
    first_trace = next(iter(example["traces"].values()))
    start = first_trace["positions"][0]
    start_height = display_height(first_trace["losses"][0], example)
    axis3d.scatter(*start, start_height, facecolors="none", edgecolors="black", marker="o", s=120, zorder=9)
    axis2d.scatter(*start, facecolors="none", edgecolors="black", marker="o", s=120, zorder=6)
    handles.extend([
        Line2D([], [], color="black", marker="o", markerfacecolor="none", linestyle="none", label="старт, t=0"),
        Line2D([], [], color="black", marker="X", linestyle="none", label="последний шаг"),
        Line2D([], [], color="black", marker="*", linestyle="none", label="минимум"),
    ])

    axis3d.view_init(elev=30, azim=example.get("azimuth", -125))
    axis3d.set_box_aspect((1.3, 1, .85))
    zlabel = "ln(1 + f)" if example.get("log_height", False) else "f(w)"
    axis3d.set(xlabel="w0", ylabel="w1", zlabel=zlabel, title="Поверхность и точки каждого шага")
    axis2d.set(xlabel="w0", ylabel="w1", title="Вид сверху: линии уровня f",
               xlim=example["bounds"][0], ylim=example["bounds"][1])
    axis2d.set_aspect("equal", adjustable="box")
    axis2d.legend(handles=handles, fontsize=8, loc="best", ncol=2)
    figure.suptitle(example["title"], fontsize=14)
    colorbar = figure.colorbar(ScalarMappable(norm=time_norm, cmap=cmap),
                               ax=[axis3d, axis2d], orientation="horizontal", shrink=.7, fraction=.045, pad=.1)
    colorbar.set_label("Номер шага t: форма точки обозначает метод, цвет - момент движения")
    plt.show()
    return figure


def interactive_landscape(example):
    grid_x, grid_y, losses = surface_grid(example, count=70)
    heights = display_height(losses, example)
    max_step = max(len(trace["losses"]) - 1 for trace in example["traces"].values())
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                           column_widths=[.53, .47], horizontal_spacing=.06)
    figure.add_trace(go.Surface(x=grid_x, y=grid_y, z=heights, opacity=.3,
                                 colorscale=[[0, "#dbe3eb"], [1, "#abb8c9"]],
                                 showscale=False, hoverinfo="skip", name="поверхность"), row=1, col=1)
    levels = np.asarray(example["levels"])
    # Каждый контур строится по точному уровню f, включая неравномерный набор уровней.
    for level in levels:
        figure.add_trace(go.Contour(x=grid_x[0], y=grid_y[:, 0], z=losses, autocontour=False,
                                    contours={"start": float(level), "end": float(level), "size": 1,
                                              "coloring": "none", "showlabels": False},
                                    line={"color": "#c4cbd3", "width": 1}, showscale=False,
                                    hoverinfo="skip", showlegend=False), row=1, col=2)
    editable_traces = []
    for name, trace in example["traces"].items():
        style = METHOD_STYLE[name]
        points = trace["positions"]
        values = display_height(trace["losses"], example)
        steps = np.arange(len(points))
        data = np.column_stack([steps, trace["losses"]])
        marker = {"symbol": style["symbol"], "size": 4, "color": steps, "coloraxis": "coloraxis"}
        line = {"color": style["color"], "width": 4}
        hover = name + "<br>Шаг %{customdata[0]:.0f}<br>w0=%{x:.4f}<br>w1=%{y:.4f}<br>f=%{customdata[1]:.5g}<extra></extra>"
        editable_traces.append(len(figure.data))
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=values,
                                     mode="lines+markers", name=name, legendgroup=name,
                                     marker=marker, line=line, customdata=data, text=steps.astype(str),
                                     textposition="top center", hovertemplate=hover), row=1, col=1)
        editable_traces.append(len(figure.data))
        figure.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="lines+markers",
                                   name=name, legendgroup=name, showlegend=False,
                                   marker={**marker, "size": 6}, line={**line, "width": 2},
                                   customdata=data, text=steps.astype(str), textposition="top center",
                                   hovertemplate=hover), row=1, col=2)
        figure.add_trace(go.Scatter3d(x=[points[-1, 0]], y=[points[-1, 1]], z=[values[-1]],
                                     mode="markers", marker={"symbol": "x", "size": 6, "color": style["color"]},
                                     showlegend=False, customdata=data[-1:], hovertemplate=hover), row=1, col=1)
        figure.add_trace(go.Scatter(x=[points[-1, 0]], y=[points[-1, 1]], mode="markers",
                                   marker={"symbol": "x", "size": 10, "color": style["color"]},
                                   showlegend=False, customdata=data[-1:], hovertemplate=hover), row=1, col=2)
    minimum = np.asarray(example["minimum"])
    minimum_height = display_height(float(example["objective"](minimum)), example)
    figure.add_trace(go.Scatter3d(x=[minimum[0]], y=[minimum[1]], z=[minimum_height],
                                 mode="markers", name="минимум", marker={"symbol": "cross", "size": 6, "color": "black"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=[minimum[0]], y=[minimum[1]], mode="markers", showlegend=False,
                               marker={"symbol": "star", "size": 12, "color": "black"}, name="минимум"), row=1, col=2)
    first = next(iter(example["traces"].values()))
    start = first["positions"][0]
    start_height = display_height(first["losses"][0], example)
    figure.add_trace(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start_height], name="старт, t=0",
                                 mode="markers", marker={"symbol": "circle-open", "size": 8, "color": "black"}), row=1, col=1)
    zlabel = "ln(1 + f)" if example.get("log_height", False) else "f(w)"
    figure.update_layout(
        title=example["title"], height=640, template="plotly_white", margin={"l": 25, "r": 30, "b": 80, "t": 85},
        scene={"xaxis_title": "w0", "yaxis_title": "w1", "zaxis_title": zlabel,
               "camera": {"eye": {"x": 1.5, "y": 1.6, "z": 1.15}}, "aspectmode": "manual",
               "aspectratio": {"x": 1.3, "y": 1, "z": .85}},
        coloraxis={"colorscale": "Viridis", "cmin": 0, "cmax": max_step,
                   "colorbar": {"title": "Шаг t", "len": .7, "x": 1.01}},
        legend={"orientation": "h", "y": -.12, "x": .05},
        updatemenus=[{"type": "buttons", "direction": "right", "x": .54, "y": 1.1,
                      "buttons": [
                          {"label": "Точки шагов", "method": "restyle", "args": [{"mode": "lines+markers"}, editable_traces]},
                          {"label": "Номера шагов", "method": "restyle", "args": [{"mode": "lines+markers+text"}, editable_traces]},
                      ]}],
    )
    figure.update_xaxes(title_text="w0", range=example["bounds"][0], row=1, col=2)
    figure.update_yaxes(title_text="w1", range=example["bounds"][1], scaleanchor="x", scaleratio=1, row=1, col=2)
    return figure


def export_interactive_landscapes(examples, path="outputs/optimizer_landscapes_3d.html"):
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Шаги оптимизаторов в 3D</title>',
             '<style>body{font:16px Arial,sans-serif;background:#f3f6fa;color:#172c42;margin:0;padding:24px}main{max-width:1500px;margin:auto}section{background:white;border-radius:12px;padding:18px;margin:20px 0;box-shadow:0 2px 12px #152d4210}p{line-height:1.6}.plotly-graph-div{width:100%!important}</style></head><body><main>',
             '<h1>Как оптимизатор движется по поверхности потерь</h1>',
             '<p>Слева - 3D: вращайте поверхность мышью. Справа - вид сверху с линиями уровня. Форма маркера задает метод, цвет точки - номер шага. Наведите на любую точку, чтобы увидеть шаг и значение f; кнопка «Номера шагов» включает подписи всех точек. Крестики отмечают последние шаги.</p>']
    for index, example in enumerate(examples):
        figure = interactive_landscape(example)
        parts.append('<section id="'+html.escape(example["key"])+'">')
        parts.append('<p>'+html.escape(example["description"])+ '</p>')
        settings = []
        for name, trace in example["traces"].items():
            parameters = trace["parameters"]
            label = f"{name}: eta={parameters['lr']:g}"
            if "momentum" in parameters:
                label += f", mu={parameters['momentum']:g}"
            settings.append(label)
        parts.append('<p><strong>Настройки:</strong> '+html.escape('  |  '.join(settings))+'</p>')
        parts.append(pio.to_html(figure, full_html=False, include_plotlyjs=(index == 0),
                                 div_id="landscape-"+example["key"], config={"displaylogo": False, "responsive": True}))
        parts.append('</section>')
    parts.append('</main></body></html>')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(parts), encoding="utf-8")
    return destination
