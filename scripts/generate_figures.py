"""Generate all 7 publication-grade, ISOPE-formatted vector PDF figures (and PNG previews)
directly from verified experimental JSON/NPZ data across RL-Airfoil and 3D-wind-turbine-blade.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.interpolate import PchipInterpolator

# Paths to source repositories
REPOS_ROOT = "/home/msds/c250089/notebook/repos"
RL_AIRFOIL_DIR = os.path.join(REPOS_ROOT, "RL-Airfoil")
BLADE_3D_DIR = os.path.join(REPOS_ROOT, "3D-wind-turbine-blade")
OUT_DIR = os.path.join(REPOS_ROOT, "ISOPE2027", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Unified ISOPE publication visual styling (Times/Serif-compatible clean typography)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.9,
    "grid.color": "#dcdcdc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "#cccccc",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Unified color palette across the entire manuscript
C_TURBO    = "#2563EB"  # Royal Blue  (Vanilla / TuRBO-1)
C_RL_META  = "#D97706"  # Amber       (RL-Meta / PPO)
C_LLM_META = "#059669"  # Emerald     (LLM-Meta-TuRBO)
C_MALIBO   = "#DC2626"  # Crimson     (Paper-Exact MALIBO)
C_BASELINE = "#52525B"  # Slate Gray  (Baseline NACA0012 / NREL Phase VI)


def save_fig(fig, stem: str):
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"preview_{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[Saved] {pdf_path}")


# ==============================================================================
# Fig. 1: Unified Multi-Paradigm Meta-Optimization Architecture (2D -> 3D)
# ==============================================================================
def make_fig1_framework():
    fig, ax = plt.subplots(figsize=(12.2, 4.0), dpi=300)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def draw_box(x, y, w, h, fc, ec, title, lines, title_color="#111827", lw=1.2, title_fs=7.8, body_fs=6.55):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        hdr_h = 0.32
        hdr = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.04,rounding_size=0.09",
            facecolor=ec, edgecolor=ec, linewidth=0.5, alpha=0.16, zorder=2.5
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - hdr_h / 2 - 0.005, title,
                ha="center", va="center", fontsize=title_fs, fontweight="bold",
                color=title_color, zorder=3)
        body_h = h - hdr_h - 0.14
        step_y = body_h / max(len(lines), 1)
        y_curr = y + h - hdr_h - 0.08 - 0.5 * step_y
        for line in lines:
            ax.text(x + 0.12, y_curr, line, ha="left", va="center",
                    fontsize=body_fs, color="#1F2937", zorder=3)
            y_curr -= step_y

    def arrow(x1, y1, x2, y2, color="#374151", lw=1.25, style="-|>", rad=0.0):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style, mutation_scale=10.5,
            color=color, linewidth=lw, zorder=4
        )
        ax.add_patch(arr)

    # Column headers
    ax.text(1.52, 3.84, "1. Constrained Benchmarks", ha="center", va="center",
            fontsize=9.1, fontweight="bold", color="#111827")
    ax.text(6.02, 3.84, "2. Three Zero-Leakage Meta-Optimization Paradigms", ha="center", va="center",
            fontsize=9.1, fontweight="bold", color="#111827")
    ax.text(10.58, 3.84, "3. 10-Seed Audit & Findings", ha="center", va="center",
            fontsize=9.1, fontweight="bold", color="#111827")

    # Left Column: 2D and 3D Benchmarks
    draw_box(
        0.10, 2.04, 2.85, 1.58, "#F8FAFC", "#475569",
        "2D Airfoil Benchmark (D = 18)",
        [
            "• NACA 0012, Re = 3.5×10⁶, α = 3.5°",
            "• 18D Kulfan CST (8U+8L+LE+TE)",
            "• NeuralFoil ('xxxlarge') solver",
            "• 10 geom./spar/smooth/C_M limits",
            "• Global feas. ≈ 13.6%, NFE = 300",
        ],
        title_color="#0F172A"
    )
    draw_box(
        0.10, 0.12, 2.85, 1.58, "#F8FAFC", "#475569",
        "3D NREL Phase VI Blade (D = 63)",
        [
            "• Seq. H: R = 5.03 m, 7 m/s, 72 rpm",
            "• 3-station C¹ PCHIP + 2×6 B-Spline",
            "• 31 Full-Cosine BEM + Snel/Viterna",
            "• Monotonic c,β + thrust T ≤ 1.15 T₀",
            "• Global feas. < 10⁻⁴, Watertight STL",
        ],
        title_color="#0F172A"
    )
    arrow(1.52, 1.99, 1.52, 1.75, color="#475569", lw=1.45)
    ax.text(2.28, 1.87, "2D → 3D Scale-Up", ha="center", va="center",
            fontsize=7.0, style="italic", fontweight="bold", color="#475569")

    # Center Column: Three Meta-Optimization Paradigms (generous width 5.04)
    draw_box(
        3.50, 2.52, 5.04, 1.10, "#FFFBEB", C_RL_META,
        "Paradigm A: Pure RL-Meta (Paired CRN PPO)",
        [
            "• Offline 90-train / 20-val grid (test excluded); exact Zero-Action Identity",
            "• 2×64 Tanh Actor-Critic adapts hyperparameters of 6 base DFO solvers",
            "• Step-wise potential-based CRN reward removes cross-condition variance",
        ],
        title_color="#92400E"
    )
    draw_box(
        3.50, 1.32, 5.04, 1.10, "#ECFDF5", C_LLM_META,
        "Paradigm B: Stagnation-Adaptive LLM-Meta-TuRBO",
        [
            "• Live local Qwen3.8-27B-FP8 (post_init Step 16 + stag ≥ 10; 0 fallbacks)",
            "• Neutral physics prompt → 12-step TR multiplier + orthogonal shape modes",
            "• Analytical CAD step damping (1.0/0.5/0.25) + 30% TR-clipped injection",
        ],
        title_color="#065F46"
    )
    draw_box(
        3.50, 0.12, 5.04, 1.10, "#FEF2F2", C_MALIBO,
        "Paradigm C: Paper-Exact MALIBO (Pan et al., ICML 2024)",
        [
            "• 4×64 ELU ResFFN (d=50) + Eq. (2) LFBO + rank KS/Cov N(0,I) prior",
            "• L-BFGS Laplace MAP/Hessian TS (γ=1/3) + 70/30 early-stop residual GBDT",
            "• Strictly paper-exact 5,120-pt global uniform search (zero TR / pre-filter)",
        ],
        title_color="#991B1B"
    )

    # Right Column: Unified Audit & Scientific Findings (width 3.02)
    draw_box(
        9.08, 0.12, 3.02, 3.50, "#EFF6FF", C_TURBO,
        "Cross-Dimensional Results",
        [
            "Protocol: 10 seeds, shared n_init = 15,",
            "NFE = 300, 100% feasible optima.",
            "",
            "1. RL-Meta Stratification (2D):",
            "   6/6 positive (+11.6% grand mean):",
            "   Py-BOBYQA +47.1%, SPSA +15.8%,",
            "   CMA-ES +8.9%, TuRBO +2.9%.",
            "",
            "2. LLM-Meta-TuRBO (2D & 3D #1):",
            "   2D: 227.40 (+16.2% vs TuRBO);",
            "   3D: 6.666 kW (+9.49%, >3× faster,",
            "   +1.6% STL vol. vs +21.2% TuRBO).",
            "",
            "3. MALIBO Scaling Law:",
            "   2D: 2.2–2.6× early lead, late sat.;",
            "   3D: 0/5,120 random hits → motivates",
            "   manifold/TR-aware Meta-BO.",
        ],
        title_color="#1E3A8A", body_fs=6.55
    )

    # Connecting arrows: Left -> Center
    arrow(2.99, 3.07, 3.45, 3.07, color=C_RL_META)
    arrow(2.99, 2.58, 3.45, 2.00, color=C_LLM_META)
    arrow(2.99, 2.22, 3.45, 0.86, color=C_MALIBO)
    arrow(2.99, 1.26, 3.45, 1.74, color=C_LLM_META)
    arrow(2.99, 0.67, 3.45, 0.67, color=C_MALIBO)

    # Connecting arrows: Center -> Right
    arrow(8.58, 3.07, 9.03, 2.68, color=C_RL_META)
    arrow(8.58, 1.87, 9.03, 1.87, color=C_LLM_META)
    arrow(8.58, 0.67, 9.03, 1.06, color=C_MALIBO)

    save_fig(fig, "fig1_framework")


# ==============================================================================
# Fig. 2: 2D Six-Optimizer Pure RL-Meta vs Vanilla Convergence Curves (2x3)
# ==============================================================================
def make_fig2_2d_rlmeta():
    with open(os.path.join(RL_AIRFOIL_DIR, "results", "benchmark_results.json")) as f:
        data = json.load(f)

    # Ordered by relative RL-Meta gain to visually expose the stratification law!
    opts_ordered = ["Py-BOBYQA", "SPSA", "CMA-ES", "DE", "TuRBO", "PSO"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 4.9), dpi=300, sharex=True)
    axes_flat = axes.flatten()
    steps = np.arange(1, 301)

    for idx, opt in enumerate(opts_ordered):
        ax = axes_flat[idx]
        v_hist = np.array([r["history"][:300] for r in data["results"][opt]["Vanilla"]])
        m_hist = np.array([r["history"][:300] for r in data["results"][opt]["RL-Meta"]])

        v_mean, v_std = v_hist.mean(axis=0), v_hist.std(axis=0)
        m_mean, m_std = m_hist.mean(axis=0), m_hist.std(axis=0)
        delta = m_mean[-1] - v_mean[-1]
        pct = delta / v_mean[-1] * 100.0

        ax.plot(steps, v_mean, color=C_TURBO, lw=1.6, label=f"Vanilla ({v_mean[-1]:.1f})")
        ax.fill_between(steps, v_mean - v_std, v_mean + v_std, color=C_TURBO, alpha=0.14)

        ax.plot(steps, m_mean, color=C_RL_META, lw=1.8, label=f"RL-Meta ({m_mean[-1]:.1f})")
        ax.fill_between(steps, m_mean - m_std, m_mean + m_std, color=C_RL_META, alpha=0.18)

        panel_lbl = f"({chr(97 + idx)}) {opt}: {delta:+.1f} ({pct:+.1f}%)"
        ax.set_title(panel_lbl, fontsize=9.2, fontweight="bold", pad=4)
        ax.set_xlim(1, 300)
        ax.set_ylim(58, 244)
        ax.grid(True, alpha=0.55)
        ax.legend(loc="lower right", fontsize=7.6, handlelength=1.6, borderpad=0.35)
        if idx % 3 == 0:
            ax.set_ylabel("Best Feasible $L/D$", fontsize=9.0, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Function Evaluations (NFE)", fontsize=9.0, fontweight="bold")

    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    save_fig(fig, "fig2_2d_rlmeta_convergence")


# ==============================================================================
# Fig. 3: 2D Four-Paradigm Comparison (Convergence + Boxplot, ZERO overlap!)
# ==============================================================================
def make_fig3_2d_paradigms():
    with open(os.path.join(RL_AIRFOIL_DIR, "results", "llm_malibo_benchmark_results.json")) as f:
        data = json.load(f)

    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    labels = {
        "Vanilla-TuRBO": "TuRBO",
        "RL-Meta-TuRBO": "RL-Meta-TuRBO",
        "LLM-Meta-TuRBO": "LLM-Meta-TuRBO",
        "MALIBO": "MALIBO (Paper-Exact)",
    }
    short_labels = ["TuRBO", "RL-Meta\nTuRBO", "LLM-Meta\nTuRBO", "MALIBO"]
    colors = {
        "Vanilla-TuRBO": C_TURBO,
        "RL-Meta-TuRBO": C_RL_META,
        "LLM-Meta-TuRBO": C_LLM_META,
        "MALIBO": C_MALIBO,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.65), dpi=300,
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})
    steps = np.arange(1, 301)

    # Panel (a): Convergence trajectories
    for m in methods:
        h = np.array([r["history"][:300] for r in data["results"][m]])
        mu, sd = h.mean(axis=0), h.std(axis=0)
        lw = 2.0 if m in ("LLM-Meta-TuRBO", "MALIBO") else 1.6
        ax1.plot(steps, mu, color=colors[m], lw=lw, label=f"{labels[m]} ({mu[-1]:.1f} ± {sd[-1]:.1f})")
        ax1.fill_between(steps, mu - sd, mu + sd, color=colors[m], alpha=0.12)

    # Clean vertical line marking n_init=15 and MALIBO lead crossover ~166
    ax1.axvline(15, color="#6B7280", ls=":", lw=0.9)
    ax1.axvline(166, color=C_MALIBO, ls="--", lw=0.85, alpha=0.55)
    ax1.text(19, 64, "$n_{\\mathrm{init}}=15$", fontsize=7.5, color="#4B5563", va="bottom")
    ax1.annotate(
        "MALIBO early lead\n(NFE 16–166, 2.2–2.6× faster)",
        xy=(68, 144), xytext=(105, 92),
        fontsize=7.4, color=C_MALIBO, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_MALIBO, lw=0.9, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=C_MALIBO, alpha=0.88, lw=0.7)
    )

    ax1.set_xlim(1, 300)
    ax1.set_ylim(58, 260)
    ax1.set_xlabel("Function Evaluations (NFE)", fontweight="bold")
    ax1.set_ylabel("Best Feasible Lift-to-Drag Ratio ($L/D$)", fontweight="bold")
    ax1.set_title("(a) Mean ± SD Convergence Trajectories (10 Seeds)", fontweight="bold", pad=5)
    ax1.grid(True, alpha=0.55)
    ax1.legend(loc="upper left", fontsize=7.6, borderpad=0.35, handlelength=1.7)

    # Panel (b): Boxplot + jitter + clean mean badges
    box_data = [[r["best_score"] for r in data["results"][m]] for m in methods]
    palette = [colors[m] for m in methods]
    bp = ax2.boxplot(
        box_data, patch_artist=True, tick_labels=short_labels, widths=0.46,
        showfliers=False, medianprops=dict(color="#111827", lw=1.8), zorder=3
    )
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.50)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color("#374151")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#374151")
        cap.set_linewidth(1.0)

    rng = np.random.RandomState(42)
    for i, sc in enumerate(box_data):
        jit = rng.uniform(-0.10, 0.10, size=len(sc))
        ax2.scatter(np.full(len(sc), i + 1) + jit, sc, color=palette[i],
                    edgecolor="#111827", s=26, lw=0.6, alpha=0.92, zorder=4)
        mu_i = float(np.mean(sc))
        ax2.scatter(i + 1, mu_i, marker="D", s=34, color="#FFFFFF",
                    edgecolor=palette[i], lw=1.4, zorder=5)
        y_badge = max(sc) + 5.0
        ax2.text(
            i + 1, y_badge, f"{mu_i:.1f}",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=palette[i],
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=palette[i], lw=0.8)
        )

    ax2.set_ylim(154, 265)
    ax2.set_ylabel("Final Best $L/D$ at NFE = 300", fontweight="bold")
    ax2.set_title("(b) Final Performance Distribution (10 Seeds)", fontweight="bold", pad=5)
    ax2.grid(True, axis="y", alpha=0.55)

    fig.tight_layout(w_pad=1.2)
    save_fig(fig, "fig3_2d_paradigms_perf")


# ==============================================================================
# Fig. 4: 2D Representative Median Optimized Airfoil Shapes (Single-Column)
# ==============================================================================
def make_fig4_2d_shapes():
    sys.path.insert(0, RL_AIRFOIL_DIR)
    from src.airfoil_utils import get_naca0012_theta, theta_to_coordinates

    with open(os.path.join(RL_AIRFOIL_DIR, "results", "llm_malibo_benchmark_results.json")) as f:
        data = json.load(f)

    naca = theta_to_coordinates(get_naca0012_theta())
    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    labels = {
        "Vanilla-TuRBO": "TuRBO",
        "RL-Meta-TuRBO": "RL-Meta-TuRBO",
        "LLM-Meta-TuRBO": "LLM-Meta-TuRBO",
        "MALIBO": "MALIBO",
    }
    colors = {
        "Vanilla-TuRBO": C_TURBO,
        "RL-Meta-TuRBO": C_RL_META,
        "LLM-Meta-TuRBO": C_LLM_META,
        "MALIBO": C_MALIBO,
    }

    fig, ax = plt.subplots(figsize=(5.6, 2.45), dpi=300)
    ax.plot(naca["x_closed"], naca["y_closed"], color=C_BASELINE, ls=":", lw=1.4,
            label="NACA 0012 (65.8)", zorder=2)
    ax.fill(naca["x_closed"], naca["y_closed"], color=C_BASELINE, alpha=0.07)

    for m in methods:
        runs = data["results"][m]
        sc = [r["best_score"] for r in runs]
        rep = runs[int(np.argsort(sc)[len(sc) // 2])]
        xc = np.array(rep["coordinates"]["x_closed"])
        yc = np.array(rep["coordinates"]["y_closed"])
        lw = 1.8 if m == "LLM-Meta-TuRBO" else 1.45
        ax.plot(xc, yc, color=colors[m], lw=lw, label=f"{labels[m]} ({rep['best_score']:.1f})", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 0.145)
    ax.set_xlabel("Normalized Chord Location ($x/c$)", fontweight="bold")
    ax.set_ylabel("$y/c$", fontweight="bold")
    ax.grid(True, alpha=0.55)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, fontsize=7.0, handlelength=1.4, columnspacing=0.8, borderpad=0.3
    )

    fig.tight_layout()
    save_fig(fig, "fig4_2d_airfoil_shapes")


# ==============================================================================
# Fig. 5: 3D NREL Phase VI Performance (Convergence + Boxplot)
# ==============================================================================
def make_fig5_3d_perf():
    with open(os.path.join(BLADE_3D_DIR, "results", "benchmark_results.json")) as f:
        data = json.load(f)

    bp_kw = float(data["metadata"]["baseline_power_kW"])
    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    labels = {"TuRBO": "TuRBO-1", "LLM-Meta-TuRBO": "LLM-Meta-TuRBO", "MALIBO": "MALIBO (Paper-Exact)"}
    short_labels = ["TuRBO-1", "LLM-Meta-TuRBO", "MALIBO"]
    colors = {"TuRBO": C_TURBO, "LLM-Meta-TuRBO": C_LLM_META, "MALIBO": C_MALIBO}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.55), dpi=300,
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})
    steps = np.arange(1, 301)

    ax1.axhline(bp_kw, color=C_BASELINE, ls="--", lw=1.3, label=f"Baseline Phase VI ({bp_kw:.3f} kW)", zorder=2)

    for m in methods:
        h = np.array([r["history"][:300] for r in data["results"][m]])
        mu = h.mean(axis=0)
        sem = h.std(axis=0) / np.sqrt(len(h))
        lw = 2.0 if m == "LLM-Meta-TuRBO" else 1.7
        ax1.plot(steps, mu, color=colors[m], lw=lw, label=f"{labels[m]} ({mu[-1]:.3f} ± {h[:,-1].std():.3f} kW)", zorder=3)
        ax1.fill_between(steps, mu - sem, mu + sem, color=colors[m], alpha=0.18, zorder=2)

    # Highlight >3x speedup milestone at NFE 100
    ax1.axvline(100, color=C_LLM_META, ls=":", lw=0.95, alpha=0.75)
    ax1.scatter([100], [6.6434], color=C_LLM_META, edgecolor="#111827", s=30, zorder=5)
    ax1.annotate(
        "NFE 100: 6.643 kW\n(>3× faster than TuRBO-1 final)",
        xy=(100, 6.6434), xytext=(126, 6.45),
        fontsize=7.5, color="#065F46", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LLM_META, lw=1.0, connectionstyle="arc3,rad=0.14"),
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=C_LLM_META, alpha=0.90, lw=0.75)
    )

    ax1.set_xlim(1, 300)
    ax1.set_ylim(6.05, 6.72)
    ax1.set_xlabel("Function Evaluations (NFE)", fontweight="bold")
    ax1.set_ylabel("Best Feasible Rotor Power (kW)", fontweight="bold")
    ax1.set_title("(a) Mean ± SEM Convergence on NREL Phase VI ($D=63$)", fontweight="bold", pad=5)
    ax1.grid(True, alpha=0.55)
    ax1.legend(loc="lower right", fontsize=7.4, borderpad=0.35, handlelength=1.6)

    # Panel (b): Boxplot + jitter + badges
    box_data = [[r["best_score"] for r in data["results"][m]] for m in methods]
    palette = [colors[m] for m in methods]
    bp = ax2.boxplot(
        box_data, patch_artist=True, tick_labels=short_labels, widths=0.44,
        showfliers=False, medianprops=dict(color="#111827", lw=1.8), zorder=3
    )
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#374151")
        w.set_linewidth(1.0)

    rng = np.random.RandomState(42)
    for i, sc in enumerate(box_data):
        jit = rng.uniform(-0.08, 0.08, size=len(sc))
        ax2.scatter(np.full(len(sc), i + 1) + jit, sc, color=palette[i],
                    edgecolor="#111827", s=28, lw=0.6, alpha=0.92, zorder=4)
        mu_i = float(np.mean(sc))
        ax2.scatter(i + 1, mu_i, marker="D", s=34, color="#FFFFFF",
                    edgecolor=palette[i], lw=1.4, zorder=5)
        ax2.text(
            i + 1, max(sc) + 0.018, f"{mu_i:.3f}",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=palette[i],
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=palette[i], lw=0.8)
        )

    ax2.set_ylim(6.08, 6.77)
    ax2.set_ylabel("Final Rotor Power at NFE = 300 (kW)", fontweight="bold")
    ax2.set_title("(b) 10-Seed Robustness & Distribution", fontweight="bold", pad=5)
    ax2.grid(True, axis="y", alpha=0.55)

    fig.tight_layout(w_pad=1.2)
    save_fig(fig, "fig5_3d_blade_perf")


# ==============================================================================
# Fig. 6: 3D Blade Aerodynamic Distributions & 3-Station Airfoil Evolution
# ==============================================================================
def make_fig6_3d_aero():
    # Load 3D blade modules cleanly
    orig_cwd = os.getcwd()
    os.chdir(BLADE_3D_DIR)
    if BLADE_3D_DIR not in sys.path:
        sys.path.insert(0, BLADE_3D_DIR)
    # Remove RL-Airfoil from sys.path front to avoid 'src' collision
    sys.path = [p for p in sys.path if p != RL_AIRFOIL_DIR]
    for mod_name in list(sys.modules.keys()):
        if mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]

    import aerosandbox as asb
    from src.blade_geometry import ROTOR_RADIUS, STATION_RADII, loft_blade_3d, BEM_NUM_SECTIONS
    from src.parameterizations import BSplineBladeModel

    with open(os.path.join(BLADE_3D_DIR, "results", "benchmark_results.json")) as f:
        data = json.load(f)

    model_bs = BSplineBladeModel()
    x0 = model_bs.get_x0()
    base_det = data["metadata"]["baseline_detailed"]

    methods = ["Baseline", "TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    labels = {"Baseline": "Baseline (Phase VI)", "TuRBO": "TuRBO-1", "LLM-Meta-TuRBO": "LLM-Meta-TuRBO", "MALIBO": "MALIBO"}
    colors = {"Baseline": C_BASELINE, "TuRBO": C_TURBO, "LLM-Meta-TuRBO": C_LLM_META, "MALIBO": C_MALIBO}

    best_dets = {"Baseline": base_det}
    best_xs = {"Baseline": x0}
    for m in ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]:
        best_r = max(data["results"][m], key=lambda r: r["best_score"])
        best_dets[m] = best_r["detailed"]
        best_xs[m] = np.array(best_r["best_x"])

    fig = plt.figure(figsize=(10.3, 4.65), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85], hspace=0.40, wspace=0.28)
    ax_c  = fig.add_subplot(gs[0, 0])
    ax_tw = fig.add_subplot(gs[0, 1])
    ax_al = fig.add_subplot(gs[0, 2])
    ax_s  = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    # Top row: Chord, Twist, Effective AoA
    for m in methods:
        det = best_dets[m]
        r_frac = np.array(det["span_radii"]) / ROTOR_RADIUS
        r_d = np.linspace(r_frac[0], r_frac[-1], 240)
        c_d = PchipInterpolator(r_frac, det["span_chord"])(r_d)
        tw_d = PchipInterpolator(r_frac, det["span_twist_deg"])(r_d)
        al_d = PchipInterpolator(r_frac, det["span_alpha_deg"])(r_d)
        ls = "--" if m == "Baseline" else "-"
        lw = 1.55 if m == "Baseline" else (1.9 if m == "LLM-Meta-TuRBO" else 1.55)
        ax_c.plot(r_d, c_d, color=colors[m], ls=ls, lw=lw, label=labels[m])
        ax_tw.plot(r_d, tw_d, color=colors[m], ls=ls, lw=lw)
        ax_al.plot(r_d, al_d, color=colors[m], ls=ls, lw=lw)

    ax_c.set_title("(a) Spanwise Chord $c(r/R)$", fontweight="bold", pad=4)
    ax_c.set_xlabel("Spanwise Fraction ($r/R$)", fontweight="bold")
    ax_c.set_ylabel("Chord $c$ (m)", fontweight="bold")
    ax_c.grid(True, alpha=0.55)

    ax_tw.set_title("(b) Spanwise Twist $\\beta(r/R)$", fontweight="bold", pad=4)
    ax_tw.set_xlabel("Spanwise Fraction ($r/R$)", fontweight="bold")
    ax_tw.set_ylabel("Twist $\\beta$ (deg)", fontweight="bold")
    ax_tw.grid(True, alpha=0.55)

    ax_al.set_title("(c) Effective AoA $\\alpha(r/R)$", fontweight="bold", pad=4)
    ax_al.set_xlabel("Spanwise Fraction ($r/R$)", fontweight="bold")
    ax_al.set_ylabel("Effective AoA $\\alpha$ (deg)", fontweight="bold")
    ax_al.axhspan(6.0, 7.5, color=C_LLM_META, alpha=0.10, zorder=0)
    ax_al.annotate(
        "Optimal $C_L/C_D$\nwindow (6.0–7.5°)",
        xy=(0.56, 6.2), xytext=(0.31, 2.3),
        fontsize=7.1, color="#065F46", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LLM_META, lw=0.85),
        bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=C_LLM_META, alpha=0.9, lw=0.65)
    )
    ax_al.grid(True, alpha=0.55)

    # Bottom row: 3 control station airfoils
    beta_c = np.linspace(0.0, np.pi, 220)
    x_c = 0.5 * (1.0 - np.cos(beta_c))
    st_titles = [
        f"(d) Root ($r={STATION_RADII[0]:.2f}$ m)",
        f"(e) Mid-Span ($r={STATION_RADII[1]:.2f}$ m)",
        f"(f) Tip ($r={STATION_RADII[2]:.2f}$ m)",
    ]
    for k in range(3):
        ax = ax_s[k]
        for m in methods:
            st_k = model_bs.get_station_info(best_xs[m])[k]
            th = st_k["theta"]
            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8], lower_weights=th[8:16],
                leading_edge_weight=float(th[16]), TE_thickness=0.0
            )
            yu = k_af.upper_coordinates(x_c)[:, 1]
            yl = k_af.lower_coordinates(x_c)[:, 1]
            ls = "--" if m == "Baseline" else "-"
            lw = 1.45 if m == "Baseline" else (1.85 if m == "LLM-Meta-TuRBO" else 1.45)
            ax.plot(x_c, yu, color=colors[m], ls=ls, lw=lw)
            ax.plot(x_c, yl, color=colors[m], ls=ls, lw=lw)
        ax.set_title(st_titles[k], fontweight="bold", pad=4)
        ax.set_xlabel("$x/c$", fontweight="bold")
        if k == 0:
            ax.set_ylabel("$y/c$", fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.18, 0.19)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.55)

    handles, leg_labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, fontsize=8.4, frameon=True, handlelength=2.0)

    save_fig(fig, "fig6_3d_blade_aero")

    # ==========================================================================
    # Fig. 7: 3D Lofted Blade Orthographic Smoothness Verification (Top & Side)
    # ==========================================================================
    fig7, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(10.2, 2.75), dpi=300, sharex=True)

    for m in methods:
        X, Y, Z = loft_blade_3d(best_xs[m], n_span=BEM_NUM_SECTIONS, n_chord=60)
        r_vals = Z[:, 0]
        x_max = np.max(X, axis=1)
        x_min = np.min(X, axis=1)
        y_max = np.max(Y, axis=1)
        y_min = np.min(Y, axis=1)
        ls = "--" if m == "Baseline" else "-"
        lw = 1.5 if m == "Baseline" else (1.9 if m == "LLM-Meta-TuRBO" else 1.5)
        ax_top.plot(r_vals, x_max, color=colors[m], ls=ls, lw=lw, label=labels[m])
        ax_top.plot(r_vals, x_min, color=colors[m], ls=ls, lw=lw)
        ax_side.plot(r_vals, y_max, color=colors[m], ls=ls, lw=lw)
        ax_side.plot(r_vals, y_min, color=colors[m], ls=ls, lw=lw)

    for ax in (ax_top, ax_side):
        for sr in STATION_RADII:
            ax.axvline(sr, color="#9CA3AF", ls=":", lw=0.9)
        ax.grid(True, alpha=0.55)
        ax.set_xlabel("Spanwise Radius $r$ (m)", fontweight="bold")

    ax_top.set_ylabel("Chordwise Envelope $X$ (m)", fontweight="bold")
    ax_top.set_title("(a) Planform Top-View Envelope ($0.30c$ Pitch Axis)", fontweight="bold", pad=4)
    ax_top.legend(loc="center", bbox_to_anchor=(0.49, 0.42), ncol=2, fontsize=7.2, borderpad=0.3, handlelength=1.5, columnspacing=0.9)

    ax_side.set_ylabel("Side-View Envelope $Y$ (m)", fontweight="bold")
    ax_side.set_title("(b) Side-View Smoothness Check (Zero Undulations)", fontweight="bold", pad=4)

    fig7.tight_layout(w_pad=1.2)
    save_fig(fig7, "fig7_3d_blade_smoothness")
    os.chdir(orig_cwd)


if __name__ == "__main__":
    make_fig1_framework()
    make_fig2_2d_rlmeta()
    make_fig3_2d_paradigms()
    make_fig4_2d_shapes()
    make_fig5_3d_perf()
    make_fig6_3d_aero()
    print("All 7 ISOPE vector PDF figures generated successfully.")
