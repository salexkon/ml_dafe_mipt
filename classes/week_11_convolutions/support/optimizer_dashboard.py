"""Общий стенд оптимизаторов: реальные обновления, HTML и память состояний."""
from pathlib import Path
import copy
import html
import json
import webbrowser

import numpy as np
import torch
from torch import nn
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

METHODS = ["SGD", "momentum", "Нестеров", "AdaGrad", "RMSprop", "Adam", "AdamW", "Lion", "Muon"]
COLORS = dict(zip(METHODS, ["#4c72b0", "#da684c", "#7963ad", "#a58018", "#2990a1", "#40844d", "#bc4d84", "#855d42", "#283b50"]))
SYMBOLS = dict(zip(METHODS, ["circle", "diamond", "square", "cross", "x", "circle-open", "diamond-open", "square-open", "circle"]))


class Lion(torch.optim.Optimizer):
    """Два коэффициента сглаживания, один сохраняемый первый момент."""
    def __init__(self, params, lr=.03, betas=(.9, .99), weight_decay=0.):
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                moment = state["exp_avg"]
                direction = beta1 * moment + (1 - beta1) * parameter.grad
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction.sign(), alpha=-group["lr"])
                moment.mul_(beta2).add_(parameter.grad, alpha=1 - beta2)
        return loss


def make_settings(rates, weight_decay=0.):
    settings = {
        "SGD": (torch.optim.SGD, {"lr": rates["SGD"], "momentum": 0., "dampening": 0., "weight_decay": weight_decay}),
        "momentum": (torch.optim.SGD, {"lr": rates["momentum"], "momentum": .9, "dampening": 0., "weight_decay": weight_decay}),
        "Нестеров": (torch.optim.SGD, {"lr": rates["Нестеров"], "momentum": .9, "dampening": 0., "nesterov": True, "weight_decay": weight_decay}),
        "AdaGrad": (torch.optim.Adagrad, {"lr": rates["AdaGrad"], "lr_decay": 0., "initial_accumulator_value": 0., "eps": 1e-10, "weight_decay": weight_decay}),
        "RMSprop": (torch.optim.RMSprop, {"lr": rates["RMSprop"], "alpha": .9, "eps": 1e-8, "momentum": 0., "centered": False, "weight_decay": weight_decay}),
        "Adam": (torch.optim.Adam, {"lr": rates["Adam"], "betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "weight_decay": weight_decay}),
        "AdamW": (torch.optim.AdamW, {"lr": rates["AdamW"], "betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "weight_decay": weight_decay}),
        "Lion": (Lion, {"lr": rates["Lion"], "betas": (.9, .99), "weight_decay": weight_decay}),
        "Muon": (torch.optim.Muon, {"lr": rates["Muon"], "momentum": .95, "nesterov": True,
                  "ns_coefficients": (3.4445, -4.775, 2.0315), "ns_steps": 5, "eps": 1e-7,
                  "adjust_lr_fn": "original", "weight_decay": weight_decay}),
    }
    assert list(settings) == METHODS
    return settings


def trace_matrix_surface(objective, start, settings, steps):
    traces = {}
    for name, (optimizer_class, parameters) in settings.items():
        point = nn.Parameter(torch.diag(torch.tensor(start, dtype=torch.float64)))
        optimizer = optimizer_class([point], **parameters)
        positions = [point.detach().diagonal().numpy().copy()]
        losses = [float(objective(point.diagonal()).detach())]
        update_norms = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            objective(point.diagonal()).backward()
            previous = point.detach().clone()
            optimizer.step()
            assert torch.isfinite(point).all(), (name, step)
            assert torch.count_nonzero(point.detach() - torch.diag(point.detach().diagonal())) == 0
            update_norms.append(float((point - previous).detach().norm()))
            positions.append(point.detach().diagonal().numpy().copy())
            losses.append(float(objective(point.diagonal()).detach()))
        assert np.isfinite(losses).all(), name
        traces[name] = {"positions": np.asarray(positions), "losses": np.asarray(losses),
                        "update_norms": np.asarray(update_norms), "parameters": parameters.copy()}
    return traces


def mechanism_traces(settings):
    """Один и тот же заданный градиент; затем отдельный опыт с нулевым градиентом данных."""
    gradient_stream = np.r_[np.full(10, 10.), np.ones(90)]
    response, decay, spectra = {}, {}, {}
    generator = torch.Generator().manual_seed(31)
    left, _ = torch.linalg.qr(torch.randn(4, 3, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    matrix_gradient = left @ torch.diag(torch.tensor([10., .7, .03])) @ right.T
    for name, (optimizer_class, parameters) in settings.items():
        point = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        solver = optimizer_class([point], **parameters)
        lengths = []
        for magnitude in gradient_stream:
            point.grad = torch.diag(torch.tensor([magnitude, magnitude / 5], dtype=torch.float64))
            previous = point.detach().clone()
            solver.step()
            lengths.append(float((point - previous).detach().norm()) / parameters["lr"])
        response[name] = np.asarray(lengths)

        point = nn.Parameter(torch.diag(torch.tensor([3., 4.], dtype=torch.float64)))
        decay_parameters = {**parameters, "lr": .03, "weight_decay": .1}
        solver = optimizer_class([point], **decay_parameters)
        norms = [float(point.detach().norm())]
        for _ in range(60):
            point.grad = torch.zeros_like(point)
            solver.step()
            norms.append(float(point.detach().norm()))
        decay[name] = np.asarray(norms)

        point = nn.Parameter(torch.zeros_like(matrix_gradient))
        solver = optimizer_class([point], **parameters)
        point.grad = matrix_gradient.clone()
        solver.step()
        direction = -point.detach() / parameters["lr"]
        direction /= direction.norm().clamp_min(1e-12)
        spectra[name] = torch.linalg.svdvals(direction).numpy()
    result = {"gradient_stream": gradient_stream, "response": response, "decay": decay, "spectra": spectra}
    return result


def _scatter(figure, row, col, name, x, y, title, **kwargs):
    trace = go.Scatter(x=x, y=y, mode="lines+markers", name=name, meta={"method": name},
                       showlegend=False, line={"color": COLORS[name], "width": 2},
                       marker={"size": 4, "color": COLORS[name]},
                       hovertemplate=name + "<br>" + title + "<br>x=%{x:.5g}<br>y=%{y:.5g}<extra></extra>", **kwargs)
    figure.add_trace(trace, row=row, col=col)


def dashboard_figure(example, mechanisms):
    titles = ["Поверхность  |  вращайте мышью", "Вид сверху  |  линии уровня", "Исходная f(w), log",
              "Длина фактического обновления", "Расстояние до минимума", "Нулевой градиент данных  |  lambda=0.1",
              "Одинаковый заданный градиент", "Ответ на историю градиентов", "Спектр нормированного обновления 4x3"]
    figure = make_subplots(rows=3, cols=3, specs=[[{"type": "scene"}, {}, {}], [{}, {}, {}], [{}, {}, {}]],
                           subplot_titles=titles, horizontal_spacing=.065, vertical_spacing=.095,
                           column_widths=[.36, .32, .32])
    horizontal = np.linspace(*example["bounds"][0], 65)
    vertical = np.linspace(*example["bounds"][1], 65)
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    losses = example["objective"](np.stack([grid_x, grid_y], axis=-1))
    heights = np.log1p(losses) if example.get("log_height") else losses
    figure.add_trace(go.Surface(x=grid_x, y=grid_y, z=heights, opacity=.28,
                               colorscale=[[0, "#e4e9f0"], [1, "#9daec1"]], showscale=False, hoverinfo="skip"), row=1, col=1)
    for level in example["levels"]:
        figure.add_trace(go.Contour(x=horizontal, y=vertical, z=losses, autocontour=False,
                                   contours={"start": level, "end": level, "size": 1, "coloring": "none", "showlabels": True},
                                   line={"color": "#bac5d2", "width": 1}, showscale=False, hoverinfo="skip", showlegend=False), row=1, col=2)
    minimum = np.asarray(example["minimum"])
    for name, trace in example["traces"].items():
        points = trace["positions"]
        steps = np.arange(len(points))
        z = np.log1p(trace["losses"]) if example.get("log_height") else trace["losses"]
        hover = name + "<br>Шаг %{customdata[0]:.0f}<br>w0=%{x:.5g}<br>w1=%{y:.5g}<br>f=%{customdata[1]:.6g}<extra></extra>"
        metadata = {"method": name, "trajectory": True}
        data = np.column_stack([steps, trace["losses"]])
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=z, mode="lines+markers",
                                     name=name, meta=metadata, showlegend=False,
                                     line={"color": COLORS[name], "width": 4},
                                     marker={"color": COLORS[name], "symbol": SYMBOLS[name], "size": 3},
                                     text=steps.astype(str), customdata=data, hovertemplate=hover), row=1, col=1)
        figure.add_trace(go.Scatter3d(x=points[-1:, 0], y=points[-1:, 1], z=z[-1:], mode="markers",
                                     name=name, meta={"method": name}, showlegend=False,
                                     marker={"color": COLORS[name], "symbol": "x", "size": 6},
                                     customdata=data[-1:], hovertemplate=hover), row=1, col=1)
        figure.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="lines+markers", name=name,
                                   meta=metadata, showlegend=False, line={"color": COLORS[name], "width": 1.8},
                                   marker={"color": COLORS[name], "symbol": SYMBOLS[name], "size": 4},
                                   text=steps.astype(str), customdata=data, hovertemplate=hover), row=1, col=2)
        figure.add_trace(go.Scatter(x=points[-1:, 0], y=points[-1:, 1], mode="markers", name=name,
                                   meta={"method": name}, showlegend=False, marker={"color": COLORS[name], "symbol": "x", "size": 9},
                                   customdata=data[-1:], hovertemplate=hover), row=1, col=2)
        _scatter(figure, 1, 3, name, steps, np.maximum(trace["losses"], 1e-16), "f(w)")
        _scatter(figure, 2, 1, name, steps[1:], trace["update_norms"], "||W_t₊1 - W_t||F")
        _scatter(figure, 2, 2, name, steps, np.linalg.norm(points - minimum, axis=1), "||w - w*||2")
        _scatter(figure, 2, 3, name, np.arange(61), mechanisms["decay"][name], "||W||F; eta=0.03, lambda=0.1")
        _scatter(figure, 3, 2, name, np.arange(1, 101), mechanisms["response"][name], "||DeltaW||F / eta")
        _scatter(figure, 3, 3, name, [1, 2, 3], mechanisms["spectra"][name], "sigmaᵢ(DeltaW / ||DeltaW||F)")
    start = next(iter(example["traces"].values()))["positions"][0]
    minimum_loss = float(example["objective"](minimum))
    minimum_z = np.log1p(minimum_loss) if example.get("log_height") else minimum_loss
    figure.add_trace(go.Scatter3d(x=[minimum[0]], y=[minimum[1]], z=[minimum_z], mode="markers",
                                 marker={"color": "black", "symbol": "cross", "size": 7}, showlegend=False,
                                 hovertemplate="Минимум<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=[minimum[0], start[0]], y=[minimum[1], start[1]], mode="markers",
                               marker={"color": "black", "symbol": ["star", "circle-open"], "size": [12, 10]},
                               text=["Минимум", "Общий старт"], hovertemplate="%{text}<extra></extra>", showlegend=False), row=1, col=2)
    figure.add_trace(go.Scatter(x=np.arange(1, 101), y=mechanisms["gradient_stream"], mode="lines",
                               line={"color": "#7b8795", "width": 3}, showlegend=False,
                               hovertemplate="Общий вход a_t<br>Шаг %{x}<br>a_t=%{y}<extra></extra>"), row=3, col=1)
    for row, col, label in [(1, 3, "f(w), log"), (2, 1, "||DeltaW||F"), (2, 2, "||w - w*||2"),
                             (2, 3, "||W||F"), (3, 1, "a_t"), (3, 2, "||DeltaW||F / eta"), (3, 3, "sigmaᵢ / ||DeltaW||F")]:
        figure.update_xaxes(title_text="номер sigmaᵢ" if (row, col) == (3, 3) else "шаг", row=row, col=col)
        figure.update_yaxes(title_text=label, row=row, col=col)
    figure.update_xaxes(title_text="w0", range=example["bounds"][0], row=1, col=2)
    figure.update_yaxes(title_text="w1", range=example["bounds"][1], scaleanchor="x", scaleratio=1, row=1, col=2)
    figure.update_yaxes(type="log", row=1, col=3)
    figure.update_layout(height=1190, template="plotly_white", font={"family": "Arial", "color": "#20364d", "size": 11},
                          margin={"l": 60, "r": 25, "b": 45, "t": 45}, uirevision="optimizer-dashboard",
                          scene={"xaxis_title": "w0", "yaxis_title": "w1", "zaxis_title": "ln(1+f)" if example.get("log_height") else "f(w)",
                                 "camera": {"eye": {"x": 1.6, "y": 1.5, "z": 1.15}}, "dragmode": "orbit", "aspectmode": "cube"})
    return figure


def trajectory_figure(example):
    horizontal = np.linspace(*example['bounds'][0], 75)
    vertical = np.linspace(*example['bounds'][1], 75)
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    values = example['objective'](np.stack([grid_x, grid_y], axis=-1))
    heights = np.log1p(values) if example.get('log_height') else values
    figure = go.Figure(go.Surface(x=grid_x, y=grid_y, z=heights, opacity=.3,
                                 colorscale=[[0, '#e4e9f0'], [1, '#9daec1']], showscale=False, hoverinfo='skip'))
    for name, trace in example['traces'].items():
        points = trace['positions']
        steps = np.arange(len(points))
        z = np.log1p(trace['losses']) if example.get('log_height') else trace['losses']
        data = np.column_stack([steps, trace['losses']])
        hover = name + '<br>Шаг %{customdata[0]:.0f}<br>w0=%{x:.5g}<br>w1=%{y:.5g}<br>f=%{customdata[1]:.6g}<extra></extra>'
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=z, mode='lines+markers',
                                     name=name, meta={'method': name, 'trajectory': True}, showlegend=False,
                                     line={'color': COLORS[name], 'width': 5},
                                     marker={'color': COLORS[name], 'symbol': SYMBOLS[name], 'size': 4},
                                     customdata=data, text=steps.astype(str), hovertemplate=hover))
        figure.add_trace(go.Scatter3d(x=points[-1:, 0], y=points[-1:, 1], z=z[-1:], mode='markers',
                                     name=name, meta={'method': name}, showlegend=False,
                                     marker={'color': COLORS[name], 'symbol': 'x', 'size': 7},
                                     customdata=data[-1:], hovertemplate=hover))
    minimum = np.asarray(example['minimum'])
    loss = float(example['objective'](minimum))
    minimum_z = np.log1p(loss) if example.get('log_height') else loss
    first = next(iter(example['traces'].values()))
    start = first['positions'][0]
    start_z = np.log1p(first['losses'][0]) if example.get('log_height') else first['losses'][0]
    figure.add_trace(go.Scatter3d(x=[minimum[0], start[0]], y=[minimum[1], start[1]], z=[minimum_z, start_z],
                                 mode='markers', marker={'color': 'black', 'symbol': ['cross', 'circle-open'], 'size': [7, 9]},
                                 text=['Минимум', 'Общий старт'], hovertemplate='%{text}<extra></extra>', showlegend=False))
    figure.update_layout(height=800, template='plotly_white', margin={'l': 10, 'r': 10, 't': 10, 'b': 10},
                          font={'family': 'Arial', 'color': '#20364d'}, uirevision='optimizer-trajectories',
                          scene={'xaxis_title': 'w0', 'yaxis_title': 'w1',
                                 'zaxis_title': 'ln(1+f)' if example.get('log_height') else 'f(w)',
                                 'camera': {'eye': {'x': 1.6, 'y': 1.5, 'z': 1.15}}, 'dragmode': 'orbit', 'aspectmode': 'cube'})
    return figure


def export_dashboard(examples, mechanisms=None, path="outputs/optimizer_landscapes_3d.html", trajectory_only=False):
    figures = {e["key"]: json.loads(pio.to_json(trajectory_figure(e) if trajectory_only else dashboard_figure(e, mechanisms))) for e in examples}
    info = {e["key"]: {"description": e["description"], "parameters": {n: t["parameters"] for n, t in e["traces"].items()}} for e in examples}
    options = ''.join(f'<option value="{html.escape(e["key"])}">{html.escape(e["title"])}</option>' for e in examples)
    checks = ''.join(f'<label style="--method:{COLORS[n]}"><input type="checkbox" data-method="{html.escape(n)}" checked>{html.escape(n)}</label>' for n in METHODS)
    encoded = json.dumps({"figures": figures, "info": info}, ensure_ascii=False).replace('</', '<\\/')
    page = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Оптимизаторы  |  общая лаборатория</title><style>
body{margin:0;background:#f1f5f9;color:#20364d;font:15px Arial,sans-serif}main{max-width:1800px;margin:auto;padding:18px}
h1{font-size:25px;margin:0 0 12px}.controls,.card{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:14px;margin-bottom:12px}
.toolbar,.methods{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.methods{margin-top:12px}label{padding:7px 10px;border-left:4px solid var(--method);background:#f7f9fc;border-radius:5px;cursor:pointer}
input{margin-right:7px;accent-color:var(--method)}button,select{font:inherit;padding:7px 12px;border:1px solid #b8c7d8;border-radius:6px;background:white;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #90b4e5}p{line-height:1.5;margin:9px 0}#plot{min-height:1190px}#parameters{font-size:13px;line-height:1.6;white-space:pre-wrap}
</style><script>__PLOTLY__</script></head><body><main>
<h1>Как оптимизаторы меняют веса</h1><div class="controls"><div class="toolbar">
<select id="geometry" aria-label="Поверхность">__OPTIONS__</select><button id="all">Все методы</button><button id="none">Скрыть методы</button>
<button id="camera">Сбросить ракурс</button><label><input id="focus3d" type="checkbox">Крупно 3D</label><label><input id="numbers" type="checkbox">Номера шагов</label></div>
<div class="methods">__CHECKS__</div><p id="description"></p>
<p>Вращайте 3D мышью; наведите на точку для номера шага и значения f. Крестики - последние шаги. Флажок метода управляет всеми его линиями и точками.</p>
<details><summary>Параметры методов на выбранной поверхности</summary><div id="parameters"></div></details></div>
<div class="card"><div id="plot"></div></div>
<p>Две нижние строки: длина шага, расстояние до минимума и три отдельных опыта. Для заданного градиента G_t=diag(a_t,a_t/5) вычисляется ответ оптимизатора без оптимизации поверхности. В опыте weight decay eta=0.03, lambda=0.1 и градиент данных нулевой. Спектр показан после первого шага по общей матрице градиента 4x3, после нормирования обновления по норме Фробениуса.</p>
</main><script id="dashboard-data" type="application/json">__DATA__</script><script>
const payload=JSON.parse(document.getElementById('dashboard-data').textContent);
const plot=document.getElementById('plot'), selector=document.getElementById('geometry');
const checks=[...document.querySelectorAll('[data-method]')];
const enabled=new Set(checks.map(c=>c.dataset.method));
let ready=Promise.resolve(), camera=null;
function visibleData(spec){return spec.data.map(t=>{const u=structuredClone(t);if(u.meta?.method)u.visible=enabled.has(u.meta.method);if(document.getElementById('focus3d').checked&&!['surface','scatter3d'].includes(u.type))u.visible=false;if(u.meta?.trajectory)u.mode=document.getElementById('numbers').checked?'lines+markers+text':'lines+markers';return u;});}
async function render(){const key=selector.value, spec=structuredClone(payload.figures[key]);if(camera)spec.layout.scene.camera=camera;
if(document.getElementById('focus3d').checked){spec.layout.scene.domain={x:[0,1],y:[0,1]};spec.layout.height=800;spec.layout.annotations=[];Object.keys(spec.layout).filter(k=>/^[xy]axis\d*$/.test(k)).forEach(k=>spec.layout[k].visible=false);plot.style.minHeight='800px';}else{plot.style.minHeight='1190px';}
document.getElementById('description').textContent=payload.info[key].description;
document.getElementById('parameters').textContent=Object.entries(payload.info[key].parameters).map(([n,p])=>n+': '+JSON.stringify(p)).join('\n');
Plotly.purge(plot);
await Plotly.newPlot(plot,visibleData(spec),spec.layout,{responsive:true,displaylogo:false,scrollZoom:true});
plot.on('plotly_relayout',e=>{if(e['scene.camera'])camera=e['scene.camera'];});
window.dashboardState={geometry:key,enabled:[...enabled]};}
function queueRender(){ready=ready.then(render);window.dashboardReady=ready;return ready;}
checks.forEach(c=>c.addEventListener('change',()=>{c.checked?enabled.add(c.dataset.method):enabled.delete(c.dataset.method);queueRender();}));
selector.addEventListener('change',queueRender);
document.getElementById('all').onclick=()=>{checks.forEach(c=>{c.checked=true;enabled.add(c.dataset.method);});queueRender();};
document.getElementById('none').onclick=()=>{checks.forEach(c=>{c.checked=false;enabled.delete(c.dataset.method);});queueRender();};
document.getElementById('numbers').onchange=queueRender;
document.getElementById('focus3d').onchange=queueRender;
document.getElementById('camera').onclick=()=>{camera={eye:{x:1.6,y:1.5,z:1.15},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};Plotly.relayout(plot,{'scene.camera':camera});};
window.dashboardReady=queueRender();
</script></body></html>'''
    page = page.replace('__PLOTLY__', get_plotlyjs()).replace('__OPTIONS__', options).replace('__CHECKS__', checks).replace('__DATA__', encoded)
    if trajectory_only:
        page = page.replace('<label><input id="focus3d" type="checkbox">Крупно 3D</label>',
                            '<label hidden><input id="focus3d" type="checkbox" checked>Крупно 3D</label>')
        start = page.index('<p>Две нижние строки:')
        stop = page.index('</p>', start) + len('</p>')
        page = page[:start] + page[stop:]
    destination = Path(path)
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[1] / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return destination


def display_dashboard(path, height=1700):
    from IPython.display import HTML, display
    import ipywidgets as widgets

    destination = Path(path).resolve()
    frame = '<iframe title="Интерактивные оптимизаторы" srcdoc="' + html.escape(destination.read_text(), quote=True) + '" style="width:100%;height:' + str(height) + 'px;border:0"></iframe>'
    display(HTML(frame))
    button = widgets.Button(description="Открыть HTML в браузере", layout=widgets.Layout(width="240px"), icon="external-link")
    button.on_click(lambda event: webbrowser.open(destination.as_uri()))
    display(button)
    return button


def measure_state(settings, shape=(128, 512)):
    rows = []
    for name, (optimizer_class, parameters) in settings.items():
        point = nn.Parameter(torch.ones(*shape, dtype=torch.float32))
        solver = optimizer_class([point], **parameters)
        for _ in range(2):
            point.grad = torch.ones_like(point)
            solver.step()
        entries = []
        size = 0
        for state in solver.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    count = value.numel() * value.element_size()
                    size += count
                    entries.append(f"{key}: {tuple(value.shape)}, {str(value.dtype)}, {count:,} байт")
        rows.append({"метод": name, "байт": size, "MiB": size / 2**20, "состояния": '<br>'.join(entries) or 'Нет сохраняемых тензоров',
                     "на один вес, байт": size / point.numel()})
    return rows


def memory_figure(rows):
    explanations = {"SGD": "нет состояния", "momentum": "1P: инерция", "Нестеров": "1P: инерция",
                    "AdaGrad": "1P: сумма квадратов + счётчик", "RMSprop": "1P: среднее квадратов + счётчик",
                    "Adam": "2P: первый и второй моменты + счётчик", "AdamW": "2P: первый и второй моменты + счётчик",
                    "Lion": "1P: первый момент", "Muon": "1P: матричная инерция"}
    figure = go.Figure()
    for row in rows:
        name = row['метод']
        figure.add_trace(go.Bar(x=[row['MiB']], y=[name], orientation='h', marker_color=COLORS[name], showlegend=False,
                                text=[f"{row['MiB']:.6f} MiB  |  {explanations[name]}" if row['байт'] else ""], textposition='outside', cliponaxis=False,
                                hovertext=[row['состояния']], hovertemplate=name+'<br>%{hovertext}<extra></extra>'))
        if row['байт'] == 0:
            figure.add_trace(go.Scatter(x=[0], y=[name], mode='markers', marker={'color': COLORS[name], 'size': 8}, showlegend=False,
                                       hovertemplate=name+'  |  нет сохраняемых тензоров<extra></extra>'))
            figure.add_annotation(x=.008, y=name, text='0 MiB  |  нет состояния', showarrow=False, xanchor='left')
    figure.update_layout(height=590, template='plotly_white', title='Дополнительное состояние  |  одна матрица 128x512, float32',
                          font={'family': 'Arial', 'color': '#20364d', 'size': 12}, margin={'l': 100, 'r': 30, 't': 65, 'b': 50},
                          xaxis={'title': 'MiB сохраняемых тензоров оптимизатора', 'range': [0, max(r['MiB'] for r in rows) * 2.9]},
                          yaxis={'autorange': 'reversed', 'categoryorder': 'array', 'categoryarray': METHODS})
    return figure


def display_memory(rows):
    from IPython.display import HTML, display

    page = pio.to_html(memory_figure(rows), include_plotlyjs=True, full_html=True, config={'displaylogo': False, 'responsive': True})
    frame = '<iframe title="Память оптимизаторов" srcdoc="' + html.escape(page, quote=True) + '" style="width:100%;height:630px;border:0"></iframe>'
    display(HTML(frame))


def build_surface_examples():
    base_rates = {"SGD": .03, "momentum": .03, "Нестеров": .03, "AdaGrad": .3,
                  "RMSprop": .03, "Adam": .15, "AdamW": .15, "Lion": .03, "Muon": .12}
    settings = make_settings(base_rates, weight_decay=0)

    def quadratic(points):
        value = (points[..., 0] ** 2 + 20 * points[..., 1] ** 2) / 2
        return value

    def narrow_valley(points):
        value = (points[..., 0] ** 2 + 25 * points[..., 1] ** 2) / 2
        return value

    def flat_basin(points):
        value = (.02 * points[..., 0] ** 2 + points[..., 1] ** 2) / 2
        return value

    def curved_valley(points):
        residual = points[..., 1] - points[..., 0] ** 2
        value = (1 - points[..., 0]) ** 2 + 20 * residual ** 2
        return value

    landscape_examples = [
        {"key": "quadratic", "title": "Разные масштабы: исходные настройки", "objective": quadratic,
         "start": (3., 2.), "minimum": (0., 0.), "steps": 100,
         "bounds": ((-3.5, 3.5), (-2.5, 2.5)), "levels": [.05, .2, .5, 1, 2, 5, 10, 20, 40, 80],
         "rates": base_rates, "description": "f=(w0^2+20w1^2)/2. Один старт (3,2), 100 обновлений, lambda=0."},
        {"key": "narrow", "title": "Узкая долина: движение между склонами", "objective": narrow_valley,
         "start": (3., 2.), "minimum": (0., 0.), "steps": 100,
         "bounds": ((-1.7, 3.5), (-2.3, 2.5)), "levels": [.05, .2, .5, 1, 2, 5, 10, 20, 40, 80],
         "rates": {**base_rates, "SGD": .075, "Adam": .22, "AdamW": .22},
         "description": "f=(w0^2+25w1^2)/2. Один старт (3,2), 100 шагов. SGD пересекает дно из-за отрицательного множителя по w1."},
        {"key": "flat", "title": "Пологая поверхность: медленное движение вдоль дна", "objective": flat_basin,
         "start": (4., 2.), "minimum": (0., 0.), "steps": 100,
         "bounds": ((-1.5, 4.7), (-2., 2.5)), "levels": [.005, .02, .05, .1, .2, .5, 1, 2, 3],
         "rates": {**base_rates, "SGD": .45, "momentum": .45, "Нестеров": .45, "Adam": .22, "AdamW": .22},
         "description": "f=(0.02w0^2+w1^2)/2. Один старт (4,2), 100 шагов. Сопоставьте loss и расстояние до (0,0)."},
        {"key": "curved", "title": "Изогнутая долина: поворот направления", "objective": curved_valley,
         "start": (-1.4, 1.6), "minimum": (1., 1.), "steps": 120, "log_height": True,
         "bounds": ((-1.8, 1.5), (-.4, 2.5)), "levels": [.02, .1, .5, 1, 2, 5, 10, 20, 40, 80],
         "rates": {"SGD": .003, "momentum": .003, "Нестеров": .003, "AdaGrad": .15, "RMSprop": .015,
                   "Adam": .07, "AdamW": .07, "Lion": .005, "Muon": .02},
         "description": "f=(1-w0)^2+20(w1-w0^2)^2. Один старт (-1.4,1.6), 120 шагов. Только высота 3D преобразована в ln(1+f)."},
    ]
    for example in landscape_examples:
        example_settings = make_settings(example["rates"], weight_decay=0)
        example["traces"] = trace_matrix_surface(example["objective"], example["start"], example_settings, example["steps"])
    result = (landscape_examples, settings)
    return result
