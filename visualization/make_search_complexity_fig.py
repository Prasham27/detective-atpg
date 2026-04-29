import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

sns.set_style("white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

rng = np.random.default_rng(7)


def make_binary_tree(depth, width=8.0, prune_prob=0.0):
    """Balanced (optionally pruned) binary tree.
    Returns: {node_id: (x, y)}, [(parent, child), ...]."""
    positions = {0: (width / 2, 0)}
    edges = []
    nid = 1
    current = [(0, width / 2)]
    for d in range(1, depth + 1):
        nxt = []
        offset = width / (2 ** (d + 1))
        for parent_id, parent_x in current:
            for direction in (-1, 1):
                if prune_prob > 0 and d >= 2 and rng.random() < prune_prob:
                    continue
                cx = parent_x + direction * offset
                positions[nid] = (cx, -d)
                edges.append((parent_id, nid))
                nxt.append((nid, cx))
                nid += 1
        current = nxt
    return positions, edges


def draw_tree(ax, positions, edges, red_frac, title):
    red_edges = set()
    if red_frac > 0 and edges:
        n_red = max(1, int(len(edges) * red_frac))
        red_edges = set(rng.choice(len(edges), size=n_red, replace=False).tolist())

    for i, (a, b) in enumerate(edges):
        xs = [positions[a][0], positions[b][0]]
        ys = [positions[a][1], positions[b][1]]
        if i in red_edges:
            ax.plot(xs, ys, color="#d73027", linewidth=1.6, alpha=0.9, zorder=2)
        else:
            ax.plot(xs, ys, color="#777777", linewidth=0.8, alpha=0.55, zorder=1)

    red_nodes = {b for i, (a, b) in enumerate(edges) if i in red_edges}
    for nid, (x, y) in positions.items():
        if nid in red_nodes:
            ax.scatter([x], [y], s=24, c="#d73027",
                       edgecolors="white", linewidth=0.4, zorder=3)
        else:
            ax.scatter([x], [y], s=18, c="#2c3e50",
                       edgecolors="white", linewidth=0.4, zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_neural_net(ax, layer_sizes, title):
    n_layers = len(layer_sizes)
    h_spacing, v_spacing = 1.1, 0.45
    pos = {}
    for li, size in enumerate(layer_sizes):
        y_top = (size - 1) * v_spacing / 2
        for ni in range(size):
            pos[(li, ni)] = (li * h_spacing, y_top - ni * v_spacing)

    for li in range(n_layers - 1):
        for ni in range(layer_sizes[li]):
            for nj in range(layer_sizes[li + 1]):
                a = pos[(li, ni)]
                b = pos[(li + 1, nj)]
                ax.plot([a[0], b[0]], [a[1], b[1]],
                        color="#bbbbbb", linewidth=0.5, alpha=0.55, zorder=1)

    for (li, ni), (x, y) in pos.items():
        ax.scatter([x], [y], s=110, c="#2c3e50",
                   edgecolors="white", linewidth=1.4, zorder=2)

    x_min = -0.35
    x_max = (n_layers - 1) * h_spacing + 0.35
    arrow = FancyArrowPatch((x_min, 0), (x_max, 0),
                            arrowstyle="-|>", mutation_scale=32,
                            color="#2ca02c", linewidth=5.0, zorder=4)
    ax.add_patch(arrow)

    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    y_pad = max(layer_sizes) * v_spacing / 2 + 0.4
    ax.set_ylim(-y_pad, y_pad)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)

# 1. D-Algorithm - deep, dense, heavy backtracking
pos1, edg1 = make_binary_tree(depth=6, width=8.0)
draw_tree(axes[0], pos1, edg1, red_frac=0.55,
          title="D-Algorithm\n(decisions on internal lines - exponential)")

# 2. PODEM - shallower (PI-only decisions)
pos2, edg2 = make_binary_tree(depth=4, width=8.0)
draw_tree(axes[1], pos2, edg2, red_frac=0.28,
          title="PODEM\n(decisions on PIs only - smaller search)")

# 3. FAN - pruned tree, headline early stopping
pos3, edg3 = make_binary_tree(depth=4, width=8.0, prune_prob=0.35)
draw_tree(axes[2], pos3, edg3, red_frac=0.15,
          title="FAN\n(headline pruning, early stopping)")

# 4. ATPP (Ours) - NN with single forward-pass arrow
draw_neural_net(axes[3], layer_sizes=(4, 6, 6, 1),
                title="ATPP - Ours\n(O(1) forward pass, no backtracking)")

plt.savefig("atpg_search_complexity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: atpg_search_complexity.png")
