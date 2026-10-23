"""Historical styles for Figures 4 and 5."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

def plot_is_size_mean_max_lines_shaded_onecolumn(
    res: dict,
    name: str,
    dir: str = ".",
    *,
    q_mean_key: str = "q_is_mean_size",
    q_max_key: str = "q_is_max_size",
    c_mean_key: str = "c_is_mean_size",
    c_max_key: str = "c_is_max_size",
    exact_mis_mean_from_log: dict | None = None,
    figsize: tuple[float, float] = (3.45, 3.4),
    x_tick_step: int = 10,
):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    x = sorted(res.keys())
    if not x:
        raise ValueError("Empty 'res' dict.")

    centers = np.array(x, dtype=float)

    def _clean(vals):
        return np.array(
            [v for v in vals if v is not None and np.isfinite(v)],
            dtype=float,
        )

    def _mean_std_by_n(key: str):
        means = []
        stds = []

        for n in x:
            vals = _clean(res[n].get(key, []))

            if vals.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)

        return np.array(means, dtype=float), np.array(stds, dtype=float)

    required_keys = [q_mean_key, q_max_key, c_mean_key, c_max_key]

    missing = [
        key for key in required_keys
        if not any(len(res[n].get(key, [])) > 0 for n in x)
    ]

    if missing:
        raise ValueError(
            "Missing data keys in res: "
            + ", ".join(missing)
            + ". If the classical keys are missing, run "
            "add_classical_mis_size_stats_to_res(...) first."
        )

    q_mean_y, q_mean_std = _mean_std_by_n(q_mean_key)
    q_max_y, q_max_std = _mean_std_by_n(q_max_key)

    c_mean_y, c_mean_std = _mean_std_by_n(c_mean_key)
    c_max_y, c_max_std = _mean_std_by_n(c_max_key)

    # Exact MIS mean manually provided from log
    if exact_mis_mean_from_log is not None:
        exact_x = np.array(
            [n for n in x if n in exact_mis_mean_from_log],
            dtype=float,
        )
        exact_y = np.array(
            [exact_mis_mean_from_log[int(n)] for n in exact_x],
            dtype=float,
        )
    else:
        exact_x = np.array([], dtype=float)
        exact_y = np.array([], dtype=float)
        
    series = [
        (
            q_mean_y,
            q_mean_std,
            "#1f77b4",
            "-",
            "o",
            r"mean quantum",
            3,
        ),
        (
            q_max_y,
            q_max_std,
            "#7b2cbf",
            "-",
            "o",
            r"max quantum",
            4,
        ),
        (
            c_mean_y,
            c_mean_std,
            "#ff7f0e",
            "--",
            "s",
            r"mean classical",
            5,
        ),
        (
            c_max_y,
            c_max_std,
            "#2ca02c",
            "--",
            "s",
            r"max classical",
            6,
        ),
    ]

    with plt.rc_context({
        "font.size": 8.3,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.7,
    }):
        fig, ax = plt.subplots(figsize=figsize)

        for y, std, color, linestyle, marker, label, z in series:
            ax.fill_between(
                centers,
                y - std,
                y + std,
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )

            ax.plot(
                centers,
                y,
                marker=marker,
                markersize=3.0,
                linewidth=1.25,
                linestyle=linestyle,
                color=color,
                label=label,
                zorder=z,
            )

        # Exact MIS mean: black dashed line
        if exact_x.size > 0:
            ax.plot(
                exact_x,
                exact_y,
                color="black",
                linestyle="--",
                linewidth=1.65,
                marker=None,
                label=r"maximum IS (exact)",
                zorder=20,
            )
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.65)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="both", which="major", length=3.8, width=0.8)

        xticks = [v for v in centers if int(v) % int(x_tick_step) == 0]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(v)) for v in xticks])

        ax.set_xlabel(r"$|V|$")
        ax.set_ylabel(r"$|S|$ on IS")

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.legend(
            loc="lower right",
            ncol=1,
            frameon=True,
            framealpha=0.88,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.35,
            borderpad=0.45,
        )

        fig.tight_layout()

        Path(dir).mkdir(parents=True, exist_ok=True)

        pdf_path = Path(dir) / f"mDS_{name}_mean_max_is_size_lines_shaded_onecolumn.pdf"
        png_path = Path(dir) / f"mDS_{name}_mean_max_is_size_lines_shaded_onecolumn.png"

        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    return ax

def plot_quantum_corrections_side_by_side(
    res: dict,
    name: str,
    dir: str = ".",
    *,
    freq_key: str = "q_corr_frac_raw_not_independent",
    sev_key: str = "q_corr_mean_conflict_edges_given_violation",
    freq_inner_label: str = r"$f_{\mathrm{non\!-\!IS}}$",
    sev_inner_label: str = r"$N_{\mathrm{viol}}$",
    color_box: str = "#9ecae1",
    figsize: tuple[float, float] = (11.2, 3.6),
    showfliers: bool = False,
):
    """Side-by-side boxplots for:"""

    x = sorted(res.keys())
    if not x:
        raise ValueError("Empty 'res' dict.")

    def _clean(vals):
        return np.array(
            [v for v in vals if v is not None and np.isfinite(v)],
            dtype=float,
        )

    freq_data = [_clean(res[n].get(freq_key, [])) for n in x]
    sev_data = [_clean(res[n].get(sev_key, [])) for n in x]

    has_freq = any(arr.size > 0 for arr in freq_data)
    has_sev = any(arr.size > 0 for arr in sev_data)

    if not has_freq and not has_sev:
        print(f"[SKIP] No data for metric_keys='{freq_key}', '{sev_key}' in this run.")
        return None

    centers = np.array(x, dtype=float)
    min_gap = float(np.min(np.diff(centers))) if len(centers) > 1 else 1.0
    box_width = 0.80 * 0.70 * min_gap

    common_kw = dict(
        patch_artist=True,
        showfliers=showfliers,
        manage_ticks=False,
        medianprops={"color": "yellow", "linewidth": 1.8},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
    )

    with plt.rc_context({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"wspace": 0.12},
        )

        def _beautify(ax):
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.9)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(
                axis="both",
                which="major",
                length=6,
                width=1.1,
                labelsize=14,
            )

            xticks = [v for v in centers if int(v) % 10 == 0]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(v)) for v in xticks])

            ax.set_xlabel(r"$|V|$", fontsize=16)
            ax.set_ylabel("")

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        def _add_inner_label(ax, text):
            ax.text(
                0.04,
                0.94,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.82,
                ),
                zorder=10,
            )

        # Left panel: frequency of non-IS samples
        _beautify(axes[0])

        if has_freq:
            bp = axes[0].boxplot(
                freq_data,
                positions=centers,
                widths=box_width,
                **common_kw,
            )

            for b in bp["boxes"]:
                b.set_facecolor(color_box)
                b.set_alpha(0.85)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
                fontsize=14,
            )

        _add_inner_label(axes[0], freq_inner_label)

        # Right panel: N_viol
        _beautify(axes[1])

        if has_sev:
            bp = axes[1].boxplot(
                sev_data,
                positions=centers,
                widths=box_width,
                **common_kw,
            )

            for b in bp["boxes"]:
                b.set_facecolor(color_box)
                b.set_alpha(0.85)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=14,
            )

        _add_inner_label(axes[1], sev_inner_label)

        Path(dir).mkdir(parents=True, exist_ok=True)

        fig.tight_layout()

        fig.savefig(
            f"{dir}/quantum_corrections_side_by_side_{name}.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            f"{dir}/quantum_corrections_side_by_side_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    return axes

