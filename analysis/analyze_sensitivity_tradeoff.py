"""
Sensitivity Trade-off Analyzer (Recall/QPS) for DynamicSegmentGraph.

This script helps you pick robust index hyperparameters (M, ef_construction,
ef_max, alpha) from a grid-search / sensitivity sweep on the DEEP dataset.

Motivation
----------
`apps/query_index.cc` prints one line per (search_ef, range bucket) containing:

  [search_ef 64] [Range ratio 1.0000%] Recall=0.8123 Latency=0.4321 ms QPS=2313.1234 ...

When you run `scripts/test_sensitivity.sh` without SEARCH_EF, each log contains
many `search_ef` values (a sweep schedule), producing a Recall–QPS curve per
range bucket for each index configuration.

Decision Rule Implemented
-------------------------
For each index configuration and each range bucket, we choose the **maximum QPS**
among points that satisfy recall constraints:

- Range 1%:  Recall >= 0.98
- Other ranges (2%, 4%, 8%, 16%, 32%, 64%): Recall >= 0.99

Then, for each range bucket, we rank configurations by that "best feasible QPS".
Finally, we summarize robustness by counting:

- **Wins**: how many ranges a configuration is rank-1
- **Top-k appearances**: how many ranges a configuration appears in the top-k

Inputs
------
- A directory of `.log` files produced by `apps/query_index` (recommended:
  `logs/static/deep/sensitivity/search/`).
- Filenames are expected to include tokens like:
    deep_N100000_k16_efc151_efm300_alpha1.0[...].log
  where `k` corresponds to HNSW out-degree M.

Outputs
-------
- Printed ranking tables (per range) and a global robustness summary.
- A CSV file containing per-range top-k rows and aggregated win/top-k counts.

Complexity
----------
Let F be the number of log files and L be the total number of lines across them.
- Time:  O(L) to parse + O(P log P) to sort per-range rankings
         (P = number of feasible (config, range) pairs; small in practice).
- Space: O(P) for stored best-per-(config, range) records.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET: str = "yt8m-video"
DEFAULT_DATA_SIZE: int = 100_000
DEFAULT_TOP_K: int = 3
DEFAULT_THRESHOLD_1PCT: float = 0.99
DEFAULT_THRESHOLD_OTHERS: float = 0.98


@dataclass(frozen=True)
class IndexConfig:
    """Index configuration parsed from the log filename."""

    dataset: str
    data_size: int
    m: int  # out-degree (called `k` in filenames, passed to build_index as -k)
    ef_construction: int
    ef_max: int
    alpha: str

    def tag(self) -> str:
        return (
            f"{self.dataset}_N{self.data_size}_k{self.m}_"
            f"efc{self.ef_construction}_efm{self.ef_max}_alpha{self.alpha}"
        )


@dataclass(frozen=True)
class BestPoint:
    """Best (max QPS) feasible point for a (config, range bucket) pair."""

    config: IndexConfig
    range_ratio: float
    threshold: float
    search_ef: int
    recall: float
    qps: float
    latency_ms: float
    log_path: Path


_FILENAME_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9_-]+)"
    r"_N(?P<N>\d+)"
    r"_k(?P<m>\d+)"
    r"_efc(?P<efc>\d+)"
    r"_efm(?P<efm>\d+)"
    r"_alpha(?P<alpha>[0-9]*\.?[0-9]+)"
    r"(?:_ef(?P<single_ef>\d+))?"
    r"(?:_(?P<ts>\d{8}_\d{6}))?"
    r"$"
)


def _repo_root_from_this_file() -> Path:
    # analysis/analyze_sensitivity_tradeoff.py -> repo root is parent of analysis/
    return Path(__file__).resolve().parents[1]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze sensitivity logs: per range bucket, pick the best QPS that "
            "meets recall thresholds, then summarize winners/top-k robustness."
        )
    )
    p.add_argument(
        "--log_root",
        type=Path,
        default=None,
        help=(
            "Directory to scan for query_index logs. "
            "Default: <repo_root>/logs/static/<dataset>/sensitivity/search"
        ),
    )
    p.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern(s) under log_root (default: '**/*.log'). Can be passed multiple times.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name filter for filenames (default: {DEFAULT_DATASET}).",
    )
    p.add_argument(
        "--data_size",
        type=int,
        default=DEFAULT_DATA_SIZE,
        help=f"Filter logs by '_N{{data_size}}_' token (default: {DEFAULT_DATA_SIZE}).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Report top-k configs per range (default: {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--threshold_1pct",
        type=float,
        default=DEFAULT_THRESHOLD_1PCT,
        help=f"Recall threshold for 1%% range (default: {DEFAULT_THRESHOLD_1PCT}).",
    )
    p.add_argument(
        "--threshold_other",
        type=float,
        default=DEFAULT_THRESHOLD_OTHERS,
        help=f"Recall threshold for other ranges (default: {DEFAULT_THRESHOLD_OTHERS}).",
    )
    p.add_argument(
        "--include",
        type=str,
        default=None,
        help="Only include log files whose relative path matches this regex.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude log files whose relative path matches this regex.",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <repo_root>/analysis/sensitivity_tradeoff.csv).",
    )
    return p


def _compile_optional_regex(expr: Optional[str]) -> Optional[re.Pattern[str]]:
    if expr is None:
        return None
    return re.compile(expr)


def _filter_paths(
    paths: Sequence[Path],
    base: Path,
    include: Optional[re.Pattern[str]],
    exclude: Optional[re.Pattern[str]],
) -> List[Path]:
    kept: List[Path] = []
    for p in paths:
        try:
            rel = str(p.relative_to(base))
        except ValueError:
            rel = str(p)
        if include is not None and include.search(rel) is None:
            continue
        if exclude is not None and exclude.search(rel) is not None:
            continue
        kept.append(p)
    return kept


def _parse_config_from_log_path(path: Path) -> Optional[IndexConfig]:
    stem = path.stem  # drop ".log"
    m = _FILENAME_RE.match(stem)
    if m is None:
        return None
    return IndexConfig(
        dataset=m.group("dataset"),
        data_size=int(m.group("N")),
        m=int(m.group("m")),
        ef_construction=int(m.group("efc")),
        ef_max=int(m.group("efm")),
        alpha=m.group("alpha"),
    )


def _threshold_for_range_ratio(
    range_ratio: float, threshold_1pct: float, threshold_other: float
) -> float:
    # Use a small tolerance instead of float == comparisons.
    if abs(range_ratio - 0.01) <= 1e-9:
        return threshold_1pct
    return threshold_other


def _write_csv(
    out_path: Path,
    per_range_rankings: Dict[float, List[BestPoint]],
    wins: Dict[IndexConfig, int],
    topk: Dict[IndexConfig, int],
    top_k: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "range_ratio,range_pct,rank,threshold,qps,recall,latency_ms,search_ef,"
        "dataset,data_size,m,ef_construction,ef_max,alpha,log_path,wins,topk_count,top_k\n"
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for ratio in sorted(per_range_rankings.keys()):
            ranked = per_range_rankings[ratio]
            for rank, bp in enumerate(ranked, start=1):
                cfg = bp.config
                f.write(
                    f"{bp.range_ratio:.6f},{bp.range_ratio * 100:.4f},"
                    f"{rank},{bp.threshold:.6f},"
                    f"{bp.qps:.6f},{bp.recall:.6f},{bp.latency_ms:.6f},{bp.search_ef},"
                    f"{cfg.dataset},{cfg.data_size},{cfg.m},{cfg.ef_construction},{cfg.ef_max},{cfg.alpha},"
                    f"\"{bp.log_path}\",{wins.get(cfg, 0)},{topk.get(cfg, 0)},{top_k}\n"
                )


def main() -> int:
    args = _build_argparser().parse_args()

    repo_root = _repo_root_from_this_file()
    dataset = str(args.dataset)
    data_size = int(args.data_size)
    top_k = max(1, int(args.top_k))

    log_root: Path
    if args.log_root is not None:
        log_root = args.log_root
    else:
        log_root = repo_root / "logs" / "static" / dataset / "sensitivity" / "search"

    out_csv: Path = (
        args.out_csv
        if args.out_csv is not None
        else (repo_root / "analysis" / "sensitivity_tradeoff.csv")
    )

    include_re = _compile_optional_regex(args.include)
    exclude_re = _compile_optional_regex(args.exclude)
    patterns: List[str] = args.pattern if args.pattern is not None else ["**/*.log"]

    try:
        from analysis.log_record_extractor import RecordExtractor
    except ImportError:
        # Allow running as `python3 analysis/analyze_sensitivity_tradeoff.py`
        # from repo root without installing anything.
        from log_record_extractor import RecordExtractor  # type: ignore

    extractor = RecordExtractor()
    range_ratios: Tuple[float, ...] = extractor.range_ratios

    if not log_root.exists():
        print(f"[analyze_sensitivity_tradeoff] log_root not found: {log_root}")
        return 1

    log_files = extractor.find_log_files(log_root, patterns=patterns)
    log_files = _filter_paths(log_files, log_root, include_re, exclude_re)

    data_size_token = f"_N{data_size}_"
    log_files = [p for p in log_files if data_size_token in p.name]

    if not log_files:
        print(f"[analyze_sensitivity_tradeoff] No log files found under: {log_root}")
        print(f"[analyze_sensitivity_tradeoff] Patterns: {patterns}")
        print(f"[analyze_sensitivity_tradeoff] data_size filter: {data_size}")
        return 1

    # Best feasible point per (config, range_ratio).
    best: Dict[Tuple[IndexConfig, float], BestPoint] = {}
    parsed_logs = 0
    skipped_logs = 0

    for log_path in log_files:
        cfg = _parse_config_from_log_path(log_path)
        if cfg is None:
            skipped_logs += 1
            continue
        if cfg.dataset != dataset or cfg.data_size != data_size:
            skipped_logs += 1
            continue

        parsed_logs += 1
        records = extractor.extract_file(log_path)
        for rec in records:
            # Apply recall threshold per range bucket.
            thr = _threshold_for_range_ratio(
                rec.range_ratio, float(args.threshold_1pct), float(args.threshold_other)
            )
            if rec.recall < thr:
                continue

            key = (cfg, rec.range_ratio)
            cur = best.get(key)
            if cur is None or rec.qps > cur.qps:
                best[key] = BestPoint(
                    config=cfg,
                    range_ratio=rec.range_ratio,
                    threshold=thr,
                    search_ef=rec.search_ef,
                    recall=rec.recall,
                    qps=rec.qps,
                    latency_ms=rec.latency_ms,
                    log_path=log_path,
                )

    if parsed_logs == 0:
        print("[analyze_sensitivity_tradeoff] No usable logs after filename parsing/filtering.")
        print(f"[analyze_sensitivity_tradeoff] scanned: {len(log_files)} file(s)")
        print(f"[analyze_sensitivity_tradeoff] skipped (unparsed or filtered): {skipped_logs}")
        return 1

    # Build per-range rankings (top_k) and robustness counts.
    wins: Dict[IndexConfig, int] = {}
    topk: Dict[IndexConfig, int] = {}
    per_range_rankings: Dict[float, List[BestPoint]] = {}

    for ratio in range_ratios:
        candidates: List[BestPoint] = []
        for (cfg, r), bp in best.items():
            if abs(r - ratio) <= 1e-12:
                candidates.append(bp)

        candidates.sort(key=lambda x: x.qps, reverse=True)
        ranked = candidates[:top_k]
        per_range_rankings[ratio] = ranked

        if ranked:
            wins[ranked[0].config] = wins.get(ranked[0].config, 0) + 1
        for bp in ranked:
            topk[bp.config] = topk.get(bp.config, 0) + 1

    # Print results.
    print("[analyze_sensitivity_tradeoff] Logs parsed:", parsed_logs)
    print("[analyze_sensitivity_tradeoff] Logs skipped:", skipped_logs)
    print(
        "[analyze_sensitivity_tradeoff] Thresholds:",
        f"1%>={float(args.threshold_1pct):.3f}, others>={float(args.threshold_other):.3f}",
    )
    print()

    for ratio in range_ratios:
        pct = int(round(ratio * 100))
        ranked = per_range_rankings.get(ratio, [])
        thr = _threshold_for_range_ratio(ratio, float(args.threshold_1pct), float(args.threshold_other))
        print(f"=== Range {pct}% (threshold recall>={thr:.3f}) ===")
        if not ranked:
            print("No feasible configurations found for this range.")
            print()
            continue
        for i, bp in enumerate(ranked, start=1):
            cfg = bp.config
            print(
                f"{i:>2}. QPS={bp.qps:>10.2f}  Recall={bp.recall:.4f}  "
                f"ef={bp.search_ef:<3d}  cfg={cfg.tag()}"
            )
        print()

    # Robustness summary.
    all_cfgs = sorted(set([cfg for (cfg, _) in best.keys()]), key=lambda c: c.tag())
    summary_rows: List[Tuple[int, int, str]] = []
    for cfg in all_cfgs:
        summary_rows.append((wins.get(cfg, 0), topk.get(cfg, 0), cfg.tag()))
    summary_rows.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    print("=== Robustness summary (sorted by wins, then top-k count) ===")
    if not summary_rows:
        print("No feasible configurations under the given thresholds.")
        return 0
    for w, t, tag in summary_rows[: min(20, len(summary_rows))]:
        print(f"Wins={w}  Top{top_k}={t}  {tag}")
    print()

    best_by_wins = summary_rows[0]
    print("Best-by-wins:", f"Wins={best_by_wins[0]}", f"Top{top_k}={best_by_wins[1]}", best_by_wins[2])

    # Write CSV (top-k per range + win/topk counts per row).
    _write_csv(out_csv, per_range_rankings, wins, topk, top_k=top_k)
    print(f"[analyze_sensitivity_tradeoff] Wrote CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




