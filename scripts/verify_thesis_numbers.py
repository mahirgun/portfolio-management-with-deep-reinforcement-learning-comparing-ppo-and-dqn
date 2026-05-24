#!/usr/bin/env python3
"""
Recompute every numeric claim in Chapter 5 of the thesis against the CSVs
in ../results/ and print a pass/fail comparison.

Exit 0 if all checks pass; 1 otherwise.

Usage:  python scripts/verify_thesis_numbers.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOL_DEFAULT = 1e-3  # 3-dp rounding slack (one unit in the last reported place)

checks: list[tuple[str, float, float, float]] = []


def check(label: str, target: float, computed: float, tol: float = TOL_DEFAULT) -> None:
    checks.append((label, float(target), float(computed), tol))


def main() -> int:
    grid   = pd.read_csv(RESULTS / "pipeline_a_validation_grid.csv")
    test_a = pd.read_csv(RESULTS / "pipeline_a_test_results.csv")
    wf     = pd.read_csv(RESULTS / "pipeline_b_test_results.csv")

    # ----- §5.1.1 Pipeline A validation grid -----
    base_v = grid[grid.algorithm.isin(["Buy_Hold", "Equal_Weight", "MktCap_Weight"])].set_index("algorithm")
    check("§5.1.1 Buy_Hold val Sharpe",     1.417, base_v.loc["Buy_Hold", "sharpe"])
    check("§5.1.1 Equal_Weight val Sharpe", 1.414, base_v.loc["Equal_Weight", "sharpe"])
    check("§5.1.1 MktCap_Weight val Sharpe",1.862, base_v.loc["MktCap_Weight", "sharpe"])

    ppo_g = grid[grid.algorithm == "PPO"]
    dqn_g = grid[grid.algorithm == "DQN"]
    check("§5.1.1 PPO grid mean Sharpe", 1.369, ppo_g.sharpe.mean())
    check("§5.1.1 PPO grid std Sharpe",  0.067, ppo_g.sharpe.std(ddof=1))
    check("§5.1.1 DQN grid mean Sharpe", 1.103, dqn_g.sharpe.mean())
    check("§5.1.1 DQN grid std Sharpe",  0.228, dqn_g.sharpe.std(ddof=1))
    check("§5.1.1 DQN/PPO grid-std ratio", 3.4, dqn_g.sharpe.std(ddof=1) / ppo_g.sharpe.std(ddof=1), tol=0.05)

    ppo_best = ppo_g[(ppo_g.timesteps == 250000)  & (ppo_g.learning_rate == 0.001)]
    dqn_best = dqn_g[(dqn_g.timesteps == 1000000) & (dqn_g.learning_rate == 0.0001)]
    check("§5.1.1 PPO best-cell mean Sharpe", 1.411, ppo_best.sharpe.mean())
    check("§5.1.1 PPO best-cell std Sharpe",  0.054, ppo_best.sharpe.std(ddof=1))
    check("§5.1.1 DQN best-cell mean Sharpe", 1.286, dqn_best.sharpe.mean())
    check("§5.1.1 DQN best-cell std Sharpe",  0.102, dqn_best.sharpe.std(ddof=1))

    ppo_cells = ppo_g.groupby(["timesteps", "learning_rate"]).sharpe.mean()
    within = int(((ppo_cells - ppo_cells.mean()).abs() <= 0.05).sum())
    check("§5.1.1 PPO cells within 0.05 of grid mean", 8, within, tol=0)

    # ----- §5.1.2 Pipeline A test set -----
    base_t = test_a[test_a.algorithm.isin(["Buy_Hold", "Equal_Weight", "MktCap_Weight"])].set_index("algorithm")
    check("§5.1.2 Buy_Hold test Sharpe",      0.162, base_t.loc["Buy_Hold", "sharpe"])
    check("§5.1.2 Equal_Weight test Sharpe",  0.203, base_t.loc["Equal_Weight", "sharpe"])
    check("§5.1.2 MktCap_Weight test Sharpe", 0.228, base_t.loc["MktCap_Weight", "sharpe"])

    ppo_t = test_a[test_a.algorithm == "PPO"]
    dqn_t = test_a[test_a.algorithm == "DQN"]
    check("§5.1.2 PPO test mean Sharpe", 0.188, ppo_t.sharpe.mean())
    check("§5.1.2 DQN test mean Sharpe", 0.128, dqn_t.sharpe.mean())
    check("§5.1.2 PPO test seed std",    0.047, ppo_t.sharpe.std(ddof=1))
    check("§5.1.2 DQN test seed std",    0.087, dqn_t.sharpe.std(ddof=1))

    check("§5.1.2 Equal_Weight ann return %", 4.48, base_t.loc["Equal_Weight", "ann_return"], tol=5e-3)
    check("§5.1.2 PPO mean ann return %",     4.09, ppo_t.ann_return.mean(),                  tol=5e-3)
    check("§5.1.2 DQN mean ann return %",     2.76, dqn_t.ann_return.mean(),                  tol=5e-3)

    check("§5.1.2 PPO mean turnover %",      0.89,  ppo_t.mean_daily_turnover.mean() * 100,                tol=5e-3)
    check("§5.1.2 Equal_Weight turnover %",  1.08,  base_t.loc["Equal_Weight",  "mean_daily_turnover"]*100, tol=5e-3)
    check("§5.1.2 MktCap_Weight turnover %", 1.27,  base_t.loc["MktCap_Weight", "mean_daily_turnover"]*100, tol=5e-3)
    check("§5.1.2 DQN mean turnover %",      17.94, dqn_t.mean_daily_turnover.mean() * 100,                tol=5e-3)

    # ----- §5.2 Pipeline B walk-forward -----
    wide = wf.pivot_table(index="window", columns="algorithm", values="sharpe")
    wide = wide[["Buy_Hold", "Equal_Weight", "MktCap_Weight", "PPO", "DQN"]]

    # §5.2.1 per-window
    check("§5.2.1 EW-PPO cross-window corr",     0.997, wide["Equal_Weight"].corr(wide["PPO"]))
    check("§5.2.1 EW-DQN cross-window corr",     0.994, wide["Equal_Weight"].corr(wide["DQN"]))
    check("§5.2.1 PPO beats EW in N windows",    2, int((wide["PPO"] > wide["Equal_Weight"]).sum()), tol=0)
    check("§5.2.1 DQN beats EW in N windows",    3, int((wide["DQN"] > wide["Equal_Weight"]).sum()), tol=0)
    check("§5.2.1 MktCap highest in N windows",  6, int((wide.idxmax(axis=1) == "MktCap_Weight").sum()), tol=0)
    check("§5.2.1 Window 5 (2022) Sharpe spread",0.113, wide.loc[5].max() - wide.loc[5].min(), tol=5e-3)
    check("§5.2.1 Window 5 (2022) all 5 strategies < 0",
          1.0, float((wide.loc[5] < 0).all()), tol=0)
    check("§5.2.1 Window 1 (2018) four of five < 0",
          4, int((wide.loc[1] < 0).sum()), tol=0)

    # §5.2.2 aggregate
    check("§5.2.2 EW mean Sharpe",       1.100, wide["Equal_Weight"].mean())
    check("§5.2.2 BH mean Sharpe",       1.042, wide["Buy_Hold"].mean())
    check("§5.2.2 MktCap mean Sharpe",   1.624, wide["MktCap_Weight"].mean())
    check("§5.2.2 PPO mean Sharpe",      1.046, wide["PPO"].mean())
    check("§5.2.2 DQN mean Sharpe",      1.084, wide["DQN"].mean())
    check("§5.2.2 PPO cross-window std", 1.378, wide["PPO"].std(ddof=1))
    check("§5.2.2 DQN cross-window std", 1.411, wide["DQN"].std(ddof=1))
    check("§5.2.2 EW cross-window std",  1.312, wide["Equal_Weight"].std(ddof=1))

    ew = wide["Equal_Weight"].values
    for algo, t_target, p_target in [
        ("PPO",            -1.059, 0.831),
        ("DQN",            -0.208, 0.578),
        ("MktCap_Weight",   3.640, 0.0075),
    ]:
        t, p2 = stats.ttest_rel(wide[algo].values, ew)
        p1 = p2 / 2 if t >= 0 else 1 - p2 / 2
        check(f"§5.2.2 {algo} vs EW t",            t_target, t,  tol=5e-3)
        check(f"§5.2.2 {algo} vs EW p (one-sided)", p_target, p1, tol=5e-4)

    t, p2 = stats.ttest_rel(wide["PPO"].values, wide["DQN"].values)
    check("§5.2.2 PPO vs DQN t (two-sided)", -0.736, t,  tol=5e-3)
    check("§5.2.2 PPO vs DQN p (two-sided)",  0.495, p2)

    # ----- Report -----
    width = max(len(c[0]) for c in checks)
    fails = 0
    for label, target, computed, tol in checks:
        ok = abs(computed - target) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label:<{width}}  target={target:>9.4f}  computed={computed:>9.4f}  diff={abs(computed-target):.4f}")
        if not ok:
            fails += 1

    print()
    print(f"Total: {len(checks)} checks, {len(checks) - fails} pass, {fails} fail")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
