"""Publication figures directly from the audited thirteen-series tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
DATA = HERE / "research_data"
EN = HERE / "manuscript/generated"
ZH = HERE / "chinese_review/generated"
ORDER = list("CGHLMNOPQRSTU")
EN_NAMES = ["Appliances", "ETTm1", "ETTm2", "GEFCom price", "GEFCom load", "OPSD load", "Household power",
            "Electricity total", "BDG2 Hog", "Steel energy", "Victoria demand", "Alabama solar", "Tetouan load"]
ZH_NAMES = ["家电能耗", "ETTm1", "ETTm2", "GEFCom电价", "GEFCom负荷", "OPSD负荷", "家庭功率",
            "电力总量", "BDG2 Hog", "钢铁能耗", "维多利亚需求", "阿拉巴马太阳能", "Tetouan负荷"]
BLUE, ORANGE, GREEN, INK = "#27658C", "#C97435", "#2C7F72", "#263746"


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name+".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (name+".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture(folder, zh):
    fig, ax = plt.subplots(figsize=(11.5, 7.1))
    ax.set(xlim=(0, 12), ylim=(0, 8.4))
    ax.axis("off")
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10", linewidth=1.3,
                                   edgecolor=INK, facecolor=color))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=15, color=INK)
    def arrow(a, b, bend=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=14, linewidth=1.45,
                                    color=INK, connectionstyle=f"arc3,rad={bend}", shrinkA=2, shrinkB=3))
    box(.35, 7.35, 11.25, .65, "已观测目标与协变量历史\n局部归一化及训练集拟合的协变量缩放" if zh else "Observed target and covariate histories\nLocal normalization and training-fitted covariate scaling", "#EAF0F3")
    box(.35, 5.8, 3.1, .9, "DLinear参考预测  d" if zh else "DLinear reference  d", "#E0F0E9")
    box(8.25, 5.8, 3.35, .9, "PatchTST参考预测  p\n与时间表征  z" if zh else "PatchTST reference  p\nand temporal state  z", "#E5EEF7")
    box(4.2, 5.8, 3.3, .9, "近期协变量上下文  c" if zh else "Recent covariate\ncontext  c", "#F6ECDE")
    arrow((1.9, 7.35), (1.9, 6.7)); arrow((5.85, 7.35), (5.85, 6.7)); arrow((9.925, 7.35), (9.925, 6.7))
    box(.35, 3.85, 4.35, 1.0, "协变量专家  E\nE = d + fE(d, c)" if zh else "Covariate expert  E\nE = d + fE(d, c)", "#E0F0E9")
    box(6.65, 3.85, 4.95, 1.0, "时序上下文专家  T\nT = p + fT(p, c, z)" if zh else "Temporal-context expert  T\nT = p + fT(p, c, z)", "#E5EEF7")
    arrow((1.9, 5.8), (1.9, 4.85))
    arrow((9.925, 5.8), (9.925, 4.85))
    arrow((5.0, 5.8), (4.1, 4.85))
    arrow((6.7, 5.8), (7.7, 4.85))
    box(3.0, 1.95, 6.0, 1.0, "二输出条件门控\n输入：E、T、c、z；权重之和为1" if zh else "Two-output conditional gate\nInputs: E, T, c, z; weights sum to one", "#EDE8F5")
    arrow((2.525, 3.85), (4.0, 2.95)); arrow((9.125, 3.85), (8.0, 2.95))
    box(3.0, .35, 6.0, .75, "加权预测\n恢复原始目标尺度" if zh else "Weighted prediction\nRestored to original target scale", "#F3F5F6")
    arrow((6.0, 1.95), (6.0, 1.1))
    save(fig, folder, "architecture")


def comparisons(folder, zh, task, summary):
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    for i, model in enumerate(("N", "PatchTST", "TimeXer", "TQNet")):
        values = task[f"vs_{model}_pct"].to_numpy()
        jitter = np.random.default_rng(41+i).uniform(-.13, .13, len(values))
        ax.scatter(values, i+jitter, s=20, color=BLUE, alpha=.55, linewidths=0)
        row = summary[summary.comparator == model].iloc[0]
        ax.errorbar(row.reduction_pct, i+.23, xerr=[[row.reduction_pct-row.lower], [row.upper-row.reduction_pct]],
                    fmt="D", color=ORANGE, markersize=6, capsize=4, linewidth=2)
    ax.axvline(0, color="#777777", linewidth=.9, linestyle="--")
    ax.set_yticks(range(4), ["数值门控N" if zh else "Numerical gate N", "PatchTST", "TimeXer", "TQNet"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE降幅（%）；正值有利于DRA" if zh else "MAE reduction (%); positive favors DRA")
    ax.grid(axis="x", color="#E2E6EA", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, folder, "comparison_distribution")


def controls(folder, zh, control):
    p = control[control.split == "test"].pivot(index=["dataset", "horizon"], columns="model", values="MAE")
    summary = pd.read_csv(DATA / "decoder_summary.csv")
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    labels = ["H+E", "H+T", "E+T"]
    for i, label in enumerate(labels):
        values = 100*(p[label]/p["H+E+T"]-1)
        ax.scatter(values, i+np.random.default_rng(51+i).uniform(-.12, .12, len(values)), s=20, alpha=.55, color=BLUE, linewidths=0)
        row = summary[(summary.split == "test") & (summary.model == label)].iloc[0]
        ax.errorbar(row.mean_MAE_increase_pct, i+.22,
                    xerr=[[row.mean_MAE_increase_pct-row.lower], [row.upper-row.mean_MAE_increase_pct]],
                    fmt="D", color=ORANGE, markersize=6, capsize=4, linewidth=2)
    ax.set_yticks(range(3), ["H + E", "H + T", "E + T  (DRA)"])
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", linestyle="--", linewidth=.9)
    ax.set_xlabel("相对H+E+T的MAE增幅（%）" if zh else "MAE increase relative to H+E+T (%)")
    ax.grid(axis="x", color="#E2E6EA", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, folder, "decoder_controls")


def heatmap(folder, zh, task):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 7.0), sharey=True, layout="constrained")
    for ax, model in zip(axes, ("N", "TQNet", "TimeXer")):
        values = task.pivot(index="dataset", columns="horizon_hours", values=f"vs_{model}_pct").reindex(ORDER)[[1, 12, 24]].to_numpy()
        picture = ax.imshow(values, cmap="RdBu", vmin=-20, vmax=30, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=14,
                    color="white" if value > 19 or value < -12 else INK)
        ax.set_xticks(range(3), ["1", "12", "24"])
        ax.set_xlabel("预测跨度（小时）" if zh else "Forecast lead (hours)")
        ax.set_title(("相对" if zh else "Versus ")+("数值门控N" if zh and model == "N" else model), fontsize=15)
    axes[0].set_yticks(range(13), ZH_NAMES if zh else EN_NAMES)
    bar = fig.colorbar(picture, ax=axes, shrink=.8, pad=.025, extend="both")
    bar.set_label("MAE降幅（%）" if zh else "MAE reduction (%)")
    save(fig, folder, "task_heatmap")


def examples(folder, zh):
    data = pd.read_csv(HERE / "qa/forecast_examples_with_observations.csv")
    fig, axes = plt.subplots(4, 1, figsize=(10.4, 8.0), gridspec_kw={"height_ratios": [2.5, 1, 2.5, 1]}, layout="constrained")
    for i, (ds, title) in enumerate((("O", "家庭功率：1小时" if zh else "Household power: 1-hour lead"),
                                    ("N", "OPSD负荷：24小时" if zh else "OPSD load: 24-hour lead"))):
        block = data[data.dataset == ds]
        x = np.arange(len(block))
        ax, error_ax = axes[2*i:2*i+2]
        ax.plot(x, block.observed, color=INK, linewidth=1.2, label="观测" if zh else "Observed")
        ax.plot(x, block.TQNet, color=ORANGE, linewidth=1.0, alpha=.9, label="TQNet")
        ax.plot(x, block.DRA, color=BLUE, linewidth=1.1, label="DRA")
        ax.set_title(title, loc="left", fontsize=14)
        ax.set_ylabel("功率（kW）" if zh and ds == "O" else "负荷（MW）" if zh else "Power (kW)" if ds == "O" else "Load (MW)")
        if i == 0:
            ax.legend(ncols=3, frameon=False, fontsize=13, loc="upper right")
        delta = abs(block.DRA-block.observed)-abs(block.TQNet-block.observed)
        error_ax.axhline(0, color="#777777", linewidth=.7)
        error_ax.fill_between(x, delta, 0, where=delta <= 0, color=BLUE, alpha=.65)
        error_ax.fill_between(x, delta, 0, where=delta > 0, color=ORANGE, alpha=.65)
        error_ax.set_ylabel("误差差值" if zh else "Error difference", fontsize=12.5)
        error_ax.set_xlabel("测试预测起点" if zh else "Test forecast origin")
        for panel in (ax, error_ax):
            panel.set_xlim(0, len(x)-1)
            panel.grid(axis="y", color="#E4E8EB", linewidth=.5)
            panel.set_axisbelow(True)
    save(fig, folder, "forecast_examples")


def main():
    task = pd.read_csv(DATA / "task_mae.csv")
    summary = pd.read_csv(DATA / "expected_primary_summary.csv")
    control = pd.read_csv(DATA / "decoder_task_mae.csv")
    for zh, folder in ((False, EN), (True, ZH)):
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.spines.top": False,
                             "axes.spines.right": False, "pdf.fonttype": 42, "axes.unicode_minus": False})
        if zh:
            font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
            from matplotlib import font_manager
            font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
            plt.rcParams["font.family"] = font.get_name()
        architecture(folder, zh)
        comparisons(folder, zh, task, summary)
        controls(folder, zh, control)
        heatmap(folder, zh, task)
        examples(folder, zh)


if __name__ == "__main__":
    main()
