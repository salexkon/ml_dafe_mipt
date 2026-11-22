"""Рисунки к коротким примерам PyTorch."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

INK = '#243b53'
BLUE = '#367c9a'
ORANGE = '#d58a32'


def plot_batches(features, targets, batch_size=2):
    """Один объект Dataset и объединение объектов в батчи."""
    values = features.detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    count, width = values.shape
    figure, axis = plt.subplots(figsize=(12, 5), layout='constrained')
    axis.set_aspect('equal')
    axis.axis('off')
    palette = plt.get_cmap('RdBu_r')
    norm = Normalize(-2, 2)

    def matrix(left, bottom, array):
        height, columns = array.shape
        for (row, column), value in np.ndenumerate(array):
            color = palette(norm(value))
            axis.add_patch(Rectangle((left + column, bottom + height - row - 1), 1, 1,
                                     facecolor=color, edgecolor='white', lw=1.5))
            axis.text(left + column + .5, bottom + height - row - .5, f'{value:g}',
                      ha='center', va='center', fontsize=12,
                      color='white' if abs(value) >= 1.5 else INK)
        axis.add_patch(Rectangle((left, bottom), columns, height, fill=False, ec='#a7b9c7', lw=1))

    matrix(0, 0, values)
    axis.text(width / 2, count + .5, 'Dataset: 6 объектов', ha='center', fontsize=14, weight='bold', color=INK)
    for index, label in enumerate(labels):
        axis.text(-.3, count - index - .5, f'i={index}', ha='right', va='center', color=INK)
        axis.text(width + .4, count - index - .5, f'y={label}', va='center', color=INK)
    for group, start in enumerate(range(0, count, batch_size)):
        batch = values[start:start + batch_size]
        left = 5 + group * 3.4
        bottom = (count - len(batch)) / 2
        matrix(left, bottom, batch)
        axis.text(left + width / 2, bottom + len(batch) + .55, f'Батч {group}', ha='center', weight='bold', color=INK)
        axis.text(left + width / 2, bottom - .6, f'X: {len(batch)}x{width}\ny: {labels[start:start + batch_size].tolist()}',
                  ha='center', va='top', color=INK, linespacing=1.6)
    axis.annotate('', (4.3, count / 2), (3.2, count / 2),
                  arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': 2})
    axis.set(xlim=(-1.1, 15), ylim=(-.6, count + 1.3))
    figure.suptitle('DataLoader объединяет строки; признаки и метки остаются на своих местах',
                   fontsize=14, color=INK)
    plt.show()
    return figure


def plot_autograd(loss, named_tensors, backward=False):
    """Схема реальных узлов loss.grad_fn.next_functions, без Graphviz."""
    named_functions = {tensor.grad_fn: (name, tensor) for name, tensor in named_tensors.items()
                       if tensor.grad_fn is not None}
    named_leaves = {id(tensor): (name, tensor) for name, tensor in named_tensors.items()
                    if tensor.is_leaf}
    nodes, edges = [], []

    def visit(function):
        if function in nodes:
            return
        nodes.append(function)
        for previous, _ in function.next_functions:
            if previous is not None:
                edges.append((previous, function))
                visit(previous)

    visit(loss.grad_fn)
    depth = {loss.grad_fn: 0}
    for _ in nodes:
        for previous, following in edges:
            if following in depth:
                depth[previous] = max(depth.get(previous, 0), depth[following] + 1)
    levels = {}
    for function in nodes:
        levels.setdefault(depth[function], []).append(function)
    maximum_depth = max(depth.values())
    positions = {}
    for level, group in levels.items():
        for index, function in enumerate(group):
            positions[function] = ((maximum_depth - level) * 2.8, (len(group) - 1) * .9 - index * 1.8)

    figure, axis = plt.subplots(figsize=(14, 4.5), layout='constrained')
    axis.axis('off')
    for previous, following in edges:
        left, right = positions[previous], positions[following]
        start = (left[0] + 1.05, left[1])
        finish = (right[0] - 1.05, right[1])
        if backward:
            start, finish = finish, start
        axis.annotate('', xy=finish, xytext=start,
                      arrowprops={'arrowstyle': '->', 'lw': 2, 'color': ORANGE if backward else '#78909c',
                                  'connectionstyle': 'arc3,rad=0.04'}, zorder=1)
    for function in nodes:
        variable = getattr(function, 'variable', None)
        if variable is not None:
            name, tensor = named_leaves.get(id(variable), ('leaf', variable))
        else:
            name, tensor = named_functions.get(function, ('operation', None))
        leaf = variable is not None
        root = function == loss.grad_fn
        x, y = positions[function]
        background = INK if leaf else '#fae6c9' if root else '#e5f0f5'
        axis.add_patch(FancyBboxPatch((x - 1.05, y - .58), 2.1, 1.16,
                                     boxstyle='round,pad=0.05,rounding_size=0.12',
                                     fc=background, ec='#b5c5d0', lw=1.1, zorder=2))
        value = f'{tensor.detach().cpu().item():g}' if tensor is not None and tensor.numel() == 1 else ''
        title = f'{name} = {value}' if value else name
        gradient = tensor.grad if tensor is not None and (tensor.is_leaf or tensor.retains_grad) else None
        grad_text = f'grad = {gradient.detach().cpu().item():g}' if gradient is not None else 'grad = None'
        label = title + '\n' + type(function).__name__
        if backward:
            label += '\n' + grad_text
        axis.text(x, y, label, ha='center', va='center', color='white' if leaf else INK,
                  fontsize=11, linespacing=1.65, zorder=3)
    axis.set(xlim=(-1.35, maximum_depth * 2.8 + 1.35), ylim=(-2, 2))
    title = 'Backward: от loss к производным' if backward else 'Forward: значения уже вычислены; связи записаны для backward'
    figure.suptitle(title, fontsize=16, color=INK, weight='bold')
    figure.supxlabel('L = (x * w + b - 8)**2  |  x=2, w=3, b=1  |  тёмные узлы - листовые тензоры',
                     fontsize=12, color=INK)
    plt.show()
    return figure
