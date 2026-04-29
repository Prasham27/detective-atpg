"""
Generates ATPG search-complexity animations for the BTech defense deck.

Outputs (all in current dir):
  d_alg.gif    - chaotic backtracking, never finishes within the loop
  podem.gif    - PI-only search, finishes at ~53 ms
  fan.gif      - headline pruning, finishes at ~21 ms
  atpp.gif     - one forward pass, finishes at ~3 ms
  combined.gif - 2x2 grid with sim-time labels (one slide-friendly file)

Each panel ticks a simulated clock so the speedup is obvious:
  ATPP:  3 ms   (1 forward pass)
  FAN:   21 ms  (~7x)
  PODEM: 53 ms  (~17.6x slower than ATPP - matches c432 result)
  D-Alg: 800+ ms still searching when the others are done

Dark theme, ~4 s loop. matplotlib FuncAnimation + PillowWriter.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'savefig.facecolor': '#0d0d0d',
    'figure.facecolor':  '#0d0d0d',
    'axes.facecolor':    '#0d0d0d',
    'text.color':        'white',
})


# ══════════════════════════════════════════════════════════════════
#  Tree builder
# ══════════════════════════════════════════════════════════════════
def make_binary_tree(depth, width=8.0, prune_prob=0.0, seed=7):
    rng = np.random.default_rng(seed)
    positions = {0: (width / 2.0, 0.0)}
    edges, children = [], {0: []}
    nid = 1
    current = [(0, width / 2.0)]
    for d in range(1, depth + 1):
        nxt = []
        offset = width / (2 ** (d + 1))
        for parent_id, parent_x in current:
            for direction in (-1, 1):
                if prune_prob > 0 and d >= 2 and rng.random() < prune_prob:
                    continue
                cx = parent_x + direction * offset
                positions[nid] = (cx, -float(d))
                edges.append((parent_id, nid))
                children.setdefault(parent_id, []).append(nid)
                children[nid] = []
                nxt.append((nid, cx))
                nid += 1
        current = nxt
    leaves = [n for n in positions if not children.get(n)]
    return positions, edges, children, leaves


def random_path(children, rng, start=0, max_depth=None):
    cur, path = start, [start]
    while children.get(cur) and (max_depth is None or len(path) <= max_depth):
        cur = int(rng.choice(children[cur]))
        path.append(cur)
    return path


# ══════════════════════════════════════════════════════════════════
#  Tree panel  (D-Alg / PODEM / FAN)
# ══════════════════════════════════════════════════════════════════
def init_tree_panel(ax, depth, width, prune_prob, title, *,
                    finish_frame=None, finish_time_ms=None,
                    ms_per_frame=12.0, rng_seed=42, fontsize=14):
    positions, edges, children, leaves = make_binary_tree(
        depth=depth, width=width, prune_prob=prune_prob, seed=7)

    edge_lines = {}
    for (a, b) in edges:
        line, = ax.plot([positions[a][0], positions[b][0]],
                        [positions[a][1], positions[b][1]],
                        color='#444444', linewidth=0.7, alpha=0.5, zorder=1)
        edge_lines[(a, b)] = line

    node_scatters = {}
    for nid, (x, y) in positions.items():
        node_scatters[nid] = ax.scatter([x], [y], s=14, c='#888888',
                                        edgecolors='#0d0d0d',
                                        linewidth=0.4, zorder=3)

    ax.set_title(title, fontsize=fontsize, fontweight='bold',
                 color='white', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    status_text = ax.text(
        0.97, 0.97, 'SEARCHING  0 ms', transform=ax.transAxes,
        ha='right', va='top', fontsize=max(fontsize - 2, 10),
        fontweight='bold', color='#ff7a7a', zorder=10,
        bbox=dict(boxstyle='round,pad=0.32', facecolor='#0d0d0d',
                  edgecolor='#ff7a7a', linewidth=0.8, alpha=0.85))

    return {
        'kind':            'tree',
        'depth':           depth,
        'children':        children,
        'edge_lines':      edge_lines,
        'node_scatters':   node_scatters,
        'edges':           edges,
        'rng':             np.random.default_rng(rng_seed),
        'heads':           [],
        'edge_intensity':  {e: 0.0 for e in edges},
        'dead_nodes':      set(),
        'headline_nodes':  set(),
        'winning_path':    None,
        'finish_frame':    finish_frame,
        'finish_time_ms':  finish_time_ms,
        'ms_per_frame':    ms_per_frame,
        'status_text':     status_text,
    }


def _reset_tree_panel(p):
    p['heads'] = []
    for e in p['edge_intensity']:
        p['edge_intensity'][e] = 0.0
    p['dead_nodes'] = set()
    p['headline_nodes'] = set()
    p['winning_path'] = None


def step_tree_panel(p, frame, n_heads, headline_mode):
    if frame == 0:
        _reset_tree_panel(p)

    rng, depth, children = p['rng'], p['depth'], p['children']
    finished = p['finish_frame'] is not None and frame >= p['finish_frame']

    if finished and p['winning_path'] is None:
        if headline_mode:
            p['winning_path'] = random_path(children, rng, max_depth=2)
        else:
            p['winning_path'] = random_path(children, rng)
        p['heads'] = []

    if not finished:
        while len(p['heads']) < n_heads:
            if headline_mode:
                stop = int(rng.integers(1, min(3, depth) + 1))
                path = random_path(children, rng, max_depth=stop)
            else:
                path = random_path(children, rng)
            p['heads'].append({'path': path, 'step': 0})

        alive = []
        for h in p['heads']:
            h['step'] += 1
            if h['step'] < len(h['path']):
                a = h['path'][h['step'] - 1]
                b = h['path'][h['step']]
                if (a, b) in p['edge_intensity']:
                    p['edge_intensity'][(a, b)] = 1.0
                alive.append(h)
            else:
                final = h['path'][-1]
                (p['headline_nodes'] if headline_mode
                 else p['dead_nodes']).add(final)
        p['heads'] = alive

        for e in p['edge_intensity']:
            p['edge_intensity'][e] *= 0.86

    win_edges = set()
    win_nodes = set()
    if p['winning_path']:
        wp = p['winning_path']
        win_edges = {(wp[i], wp[i + 1]) for i in range(len(wp) - 1)}
        win_nodes = set(wp)

    artists = []
    for e, line in p['edge_lines'].items():
        if e in win_edges:
            line.set_color('#39ff7a')
            line.set_alpha(1.0)
            line.set_linewidth(2.6)
        else:
            i = p['edge_intensity'][e]
            if i > 0.05:
                r = 0.27 + 0.73 * i
                g = 0.27 * (1 - i) + 0.18 * i
                b = 0.27 * (1 - i) + 0.18 * i
                line.set_color((r, g, b))
                line.set_alpha(0.5 + 0.5 * i)
                line.set_linewidth(0.7 + 1.6 * i)
            else:
                line.set_color('#444444')
                line.set_alpha(0.5)
                line.set_linewidth(0.7)
        artists.append(line)

    for nid, sc in p['node_scatters'].items():
        if nid in win_nodes:
            sc.set_color('#39ff7a'); sc.set_sizes([46])
        elif nid in p['dead_nodes']:
            sc.set_color('#ff2d2d'); sc.set_sizes([34])
        elif nid in p['headline_nodes']:
            sc.set_color('#39ff7a'); sc.set_sizes([42])
        else:
            sc.set_color('#888888'); sc.set_sizes([14])
        artists.append(sc)

    if not finished:
        for h in p['heads']:
            cur = h['path'][min(h['step'], len(h['path']) - 1)]
            if (cur not in p['dead_nodes']
                    and cur not in p['headline_nodes']
                    and cur not in win_nodes):
                p['node_scatters'][cur].set_color('#ff9090')
                p['node_scatters'][cur].set_sizes([32])

    if finished:
        p['status_text'].set_text(f"DONE   {p['finish_time_ms']:.0f} ms")
        p['status_text'].set_color('#39ff7a')
        bbox = p['status_text'].get_bbox_patch()
        if bbox is not None:
            bbox.set_edgecolor('#39ff7a')
    else:
        elapsed = frame * p['ms_per_frame']
        p['status_text'].set_text(f"SEARCHING   {elapsed:.0f} ms")
        p['status_text'].set_color('#ff7a7a')
        bbox = p['status_text'].get_bbox_patch()
        if bbox is not None:
            bbox.set_edgecolor('#ff7a7a')
    artists.append(p['status_text'])
    return artists


# ══════════════════════════════════════════════════════════════════
#  ATPP panel
# ══════════════════════════════════════════════════════════════════
def init_atpp_panel(ax, n_frames, title, *,
                    flow_frames=8, finish_time_ms=3.0,
                    fontsize=14):
    layer_sizes = (4, 6, 6, 1)
    h_spacing, v_spacing = 1.1, 0.45
    pos = {}
    for li, size in enumerate(layer_sizes):
        y_top = (size - 1) * v_spacing / 2
        for ni in range(size):
            pos[(li, ni)] = (li * h_spacing, y_top - ni * v_spacing)

    edges = [((li, ni), (li + 1, nj))
             for li in range(len(layer_sizes) - 1)
             for ni in range(layer_sizes[li])
             for nj in range(layer_sizes[li + 1])]

    edge_lines = {}
    for (a, b) in edges:
        line, = ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color='#444444', linewidth=0.6, alpha=0.5, zorder=1)
        edge_lines[(a, b)] = line

    node_scatters = {}
    for k, (x, y) in pos.items():
        node_scatters[k] = ax.scatter([x], [y], s=110, c='#888888',
                                      edgecolors='#0d0d0d',
                                      linewidth=1.2, zorder=2)

    x_min = -0.35
    x_max = (len(layer_sizes) - 1) * h_spacing + 0.35

    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    y_pad = max(layer_sizes) * v_spacing / 2 + 0.4
    ax.set_ylim(-y_pad, y_pad)
    ax.set_title(title, fontsize=fontsize, fontweight='bold',
                 color='white', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    status_text = ax.text(
        0.97, 0.97, 'RUNNING  0.0 ms', transform=ax.transAxes,
        ha='right', va='top', fontsize=max(fontsize - 2, 10),
        fontweight='bold', color='#3ad17a', zorder=10,
        bbox=dict(boxstyle='round,pad=0.32', facecolor='#0d0d0d',
                  edgecolor='#3ad17a', linewidth=0.8, alpha=0.85))

    return {
        'kind':            'atpp',
        'ax':              ax,
        'pos':             pos,
        'edge_lines':      edge_lines,
        'node_scatters':   node_scatters,
        'x_min':           x_min,
        'x_max':           x_max,
        'flow_frames':     flow_frames,
        'finish_time_ms':  finish_time_ms,
        'arrow':           {'patch': None},
        'status_text':     status_text,
    }


def step_atpp_panel(p, frame):
    flow_frames = p['flow_frames']
    t = frame / max(flow_frames - 1, 1) if frame < flow_frames else 1.0
    wave_x = p['x_min'] + t * (p['x_max'] - p['x_min'])
    finished = frame >= flow_frames

    if p['arrow']['patch'] is not None:
        p['arrow']['patch'].remove()
        p['arrow']['patch'] = None
    if t > 0.02:
        p['arrow']['patch'] = FancyArrowPatch(
            (p['x_min'], 0), (wave_x, 0),
            arrowstyle='-|>', mutation_scale=34,
            color='#39ff7a', linewidth=5.5, zorder=4, alpha=0.95)
        p['ax'].add_patch(p['arrow']['patch'])

    for (a, b), line in p['edge_lines'].items():
        mid_x = (p['pos'][a][0] + p['pos'][b][0]) / 2
        d = wave_x - mid_x
        if -0.15 < d < 0.4:
            line.set_color('#39ff7a'); line.set_alpha(0.9); line.set_linewidth(1.4)
        elif d >= 0.4:
            line.set_color('#258a45'); line.set_alpha(0.6); line.set_linewidth(0.8)
        else:
            line.set_color('#444444'); line.set_alpha(0.5); line.set_linewidth(0.6)

    for k, sc in p['node_scatters'].items():
        d = wave_x - p['pos'][k][0]
        if -0.2 < d < 0.25:
            sc.set_color('#39ff7a'); sc.set_sizes([180])
        elif d >= 0.25:
            sc.set_color('#3ad17a'); sc.set_sizes([130])
        else:
            sc.set_color('#888888'); sc.set_sizes([110])

    if finished:
        p['status_text'].set_text(f"DONE   {p['finish_time_ms']:.0f} ms")
    else:
        scale = p['finish_time_ms'] / max(flow_frames, 1)
        p['status_text'].set_text(f"RUNNING   {frame * scale:.1f} ms")

    return list(p['edge_lines'].values()) + list(p['node_scatters'].values()) + [p['status_text']]


# ══════════════════════════════════════════════════════════════════
#  Single-panel GIF
# ══════════════════════════════════════════════════════════════════
def render_single(filename, kind, n_frames, fps, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    if kind == 'tree':
        panel = init_tree_panel(
            ax, depth=kwargs['depth'], width=kwargs['width'],
            prune_prob=kwargs['prune_prob'], title=kwargs['title'],
            finish_frame=kwargs.get('finish_frame'),
            finish_time_ms=kwargs.get('finish_time_ms'),
            ms_per_frame=kwargs.get('ms_per_frame', 12.0),
            rng_seed=kwargs.get('seed', 42))
        n_heads  = kwargs['n_heads']
        headline = kwargs.get('headline_mode', False)
        update_fn = lambda f: step_tree_panel(panel, f, n_heads, headline)
    else:
        panel = init_atpp_panel(
            ax, n_frames, title=kwargs['title'],
            flow_frames=kwargs.get('flow_frames', 8),
            finish_time_ms=kwargs.get('finish_time_ms', 3.0))
        update_fn = lambda f: step_atpp_panel(panel, f)

    anim = animation.FuncAnimation(fig, update_fn, frames=n_frames,
                                   interval=1000 / fps, blit=False, repeat=True)
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════
#  Combined 2x2 GIF
# ══════════════════════════════════════════════════════════════════
def render_combined(filename, n_frames, fps,
                    podem_finish_frame, podem_ms,
                    fan_finish_frame,   fan_ms,
                    atpp_flow_frames,   atpp_ms,
                    dalg_ms_per_frame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.04, hspace=0.18)

    p_dalg = init_tree_panel(
        axes[0, 0], depth=6, width=8.0, prune_prob=0.0,
        title='D-Algorithm  (chaotic backtracking)',
        finish_frame=None,
        ms_per_frame=dalg_ms_per_frame,
        rng_seed=42, fontsize=13)

    p_podem = init_tree_panel(
        axes[0, 1], depth=4, width=8.0, prune_prob=0.0,
        title='PODEM  (PI-only decisions)',
        finish_frame=podem_finish_frame, finish_time_ms=podem_ms,
        ms_per_frame=podem_ms / podem_finish_frame,
        rng_seed=43, fontsize=13)

    p_fan = init_tree_panel(
        axes[1, 0], depth=4, width=8.0, prune_prob=0.35,
        title='FAN  (headline pruning)',
        finish_frame=fan_finish_frame, finish_time_ms=fan_ms,
        ms_per_frame=fan_ms / fan_finish_frame,
        rng_seed=44, fontsize=13)

    p_atpp = init_atpp_panel(
        axes[1, 1], n_frames,
        title='ATPP  -  Ours  (O(1) forward pass)',
        flow_frames=atpp_flow_frames, finish_time_ms=atpp_ms,
        fontsize=13)

    fig.suptitle('ATPG Search Complexity   -   Classical vs ML',
                 fontsize=17, fontweight='bold', color='white', y=0.985)

    def update(frame):
        artists = []
        artists += step_tree_panel(p_dalg,  frame, n_heads=3, headline_mode=False)
        artists += step_tree_panel(p_podem, frame, n_heads=1, headline_mode=False)
        artists += step_tree_panel(p_fan,   frame, n_heads=1, headline_mode=True)
        artists += step_atpp_panel(p_atpp,  frame)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False, repeat=True)
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fps      = 15
    duration = 10.0
    n_frames = int(fps * duration)            # 150 frames

    # Sim-time budgets (chosen so PODEM/ATPP ratio = 17.6x as in c432)
    # Frame budgets pace the visuals; finish_time_ms reflects real speedup.
    ATPP_MS,  ATPP_FLOW   = 3.0,  10          # done at ~0.67 s
    FAN_MS,   FAN_FRAME   = 21.0, 38          # done at ~2.5 s
    PODEM_MS, PODEM_FRAME = 53.0, 75          # done at ~5.0 s
    DALG_MS_PER_FRAME     = 6.0               # ~900 ms by end, never finishes

    # Individual GIFs (one per algorithm)
    render_single('d_alg.gif', kind='tree', n_frames=n_frames, fps=fps,
                  depth=6, width=8.0, prune_prob=0.0, n_heads=3,
                  ms_per_frame=DALG_MS_PER_FRAME,
                  title='D-Algorithm  (chaotic backtracking)')

    render_single('podem.gif', kind='tree', n_frames=n_frames, fps=fps,
                  depth=4, width=8.0, prune_prob=0.0, n_heads=1,
                  finish_frame=PODEM_FRAME, finish_time_ms=PODEM_MS,
                  ms_per_frame=PODEM_MS / PODEM_FRAME,
                  title='PODEM  (PI-only decisions)')

    render_single('fan.gif', kind='tree', n_frames=n_frames, fps=fps,
                  depth=4, width=8.0, prune_prob=0.35, n_heads=1,
                  headline_mode=True,
                  finish_frame=FAN_FRAME, finish_time_ms=FAN_MS,
                  ms_per_frame=FAN_MS / FAN_FRAME,
                  title='FAN  (headline pruning)')

    render_single('atpp.gif', kind='atpp', n_frames=n_frames, fps=fps,
                  flow_frames=ATPP_FLOW, finish_time_ms=ATPP_MS,
                  title='ATPP  -  Ours  (O(1) forward pass)')

    # 2x2 combined for the slide
    render_combined('combined.gif', n_frames=n_frames, fps=fps,
                    podem_finish_frame=PODEM_FRAME, podem_ms=PODEM_MS,
                    fan_finish_frame=FAN_FRAME, fan_ms=FAN_MS,
                    atpp_flow_frames=ATPP_FLOW, atpp_ms=ATPP_MS,
                    dalg_ms_per_frame=DALG_MS_PER_FRAME)
