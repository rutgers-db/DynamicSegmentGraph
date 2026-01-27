# 2025/12/30 Zhencan Peng: add a lightweight parser for Recall/QPS benchmark lines.

"""
Log Record Extractor for DynamicSegmentGraph Benchmark Output.

This module provides a small, dependency-light parser for the log files produced
by this repository's benchmarking binaries (e.g., `apps/query_index.cc`).

Background
----------
`scripts/run_query_index.sh` runs the `query_index` binary and tees its stdout to
`logs/static/<dataset>/search/*.log`. For each `search_ef` setting, the binary prints
one summary line per range bucket (there are 7 buckets by default):

  [search_ef 64] [Range ratio 1.0000%] Recall=0.8123 Latency=0.4321 ms QPS=2313.1234 ...

We want to extract (Recall, QPS) points for each of the 7 range buckets so we
can plot QPS vs Recall curves.

Design Goals
------------
- Robust to unrelated log lines (non-matching lines are ignored).
- Stable range bucketing: range ratios are snapped to the known 7 buckets
  (1%, 2%, 4%, 8%, 16%, 32%, 64%) with a configurable tolerance.
- Simple API and standard-library only (plotting is handled elsewhere).

Complexity
----------
Let L be the number of lines in the log file. Parsing is O(L) time and O(R)
space for R extracted records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_RANGE_RATIOS: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)

# Matches lines printed by `apps/query_index.cc`:
#   [search_ef 16] [Range ratio 1.0000%] Recall=0.1234 Latency=0.5678 ms QPS=1234.5678 ...
_QUERY_INDEX_LINE_RE = re.compile(
    r"\[search_ef\s+(?P<search_ef>\d+)\]\s+"
    r"\[Range ratio\s+(?P<range_pct>[0-9]*\.?[0-9]+)%\]\s+"
    r"Recall=(?P<recall>[0-9]*\.?[0-9]+)\s+"
    r"Latency=(?P<latency_ms>[0-9]*\.?[0-9]+)\s+ms\s+"
    r"QPS=(?P<qps>[0-9]*\.?[0-9]+)"
)


@dataclass(frozen=True)
class BenchmarkRecord:
    """One parsed benchmark point from a log line."""

    log_path: Path
    search_ef: int
    range_ratio: float  # e.g. 0.01
    recall: float
    qps: float
    latency_ms: float


class RecordExtractor:
    """
    Extract Recall/QPS benchmark points from log files.

    The extractor is intentionally strict about the expected line structure but
    forgiving about extra noise: it only consumes lines that match the known
    output format.
    """

    def __init__(
        self,
        range_ratios: Sequence[float] = DEFAULT_RANGE_RATIOS,
        range_ratio_tolerance: float = 1e-3,
    ) -> None:
        self._range_ratios: Tuple[float, ...] = tuple(range_ratios)
        self._tol: float = float(range_ratio_tolerance)

    @property
    def range_ratios(self) -> Tuple[float, ...]:
        return self._range_ratios

    def extract_file(self, log_path: Path) -> List[BenchmarkRecord]:
        """
        Parse a single log file and return a list of BenchmarkRecord.

        Args:
            log_path: Path to a `.log` file.

        Returns:
            A list of parsed records; non-matching lines are ignored.

        Time: O(L) where L is the number of lines in the file.
        Space: O(R) where R is the number of extracted records.
        """
        path = Path(log_path)
        records: List[BenchmarkRecord] = []

        # Avoid per-line exceptions for speed; decode errors are replaced.
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = self._parse_query_index_line(line, path)
                if rec is not None:
                    records.append(rec)
        return records

    def group_by_range_ratio(
        self, records: Iterable[BenchmarkRecord]
    ) -> Dict[float, List[BenchmarkRecord]]:
        """
        Group records by range ratio.

        This always returns all configured buckets (even if a bucket has no
        points) so downstream plotting can assume 7 subplots.
        """
        grouped: Dict[float, List[BenchmarkRecord]] = {r: [] for r in self._range_ratios}
        for rec in records:
            grouped.setdefault(rec.range_ratio, []).append(rec)
        return grouped

    def extract_file_grouped(
        self, log_path: Path
    ) -> Dict[float, List[BenchmarkRecord]]:
        """Convenience wrapper: extract then group by range ratio."""
        return self.group_by_range_ratio(self.extract_file(log_path))

    def find_log_files(
        self,
        log_root: Path,
        patterns: Sequence[str] = ("**/*.log",),
    ) -> List[Path]:
        """
        Recursively find log files under a root directory.

        Args:
            log_root: Root folder, e.g. `<repo>/logs`.
            patterns: Glob patterns (recursive) applied under log_root.

        Returns:
            Sorted list of file paths.
        """
        root = Path(log_root)
        files: List[Path] = []
        for pat in patterns:
            files.extend(p for p in root.glob(pat) if p.is_file())
        files = sorted(set(files))
        return files

    def _parse_query_index_line(
        self, line: str, log_path: Path
    ) -> Optional[BenchmarkRecord]:
        m = _QUERY_INDEX_LINE_RE.search(line)
        if m is None:
            return None

        search_ef = int(m.group("search_ef"))
        range_pct = float(m.group("range_pct"))
        recall = float(m.group("recall"))
        latency_ms = float(m.group("latency_ms"))
        qps = float(m.group("qps"))

        range_ratio = self._snap_range_ratio(range_pct / 100.0)

        return BenchmarkRecord(
            log_path=log_path,
            search_ef=search_ef,
            range_ratio=range_ratio,
            recall=recall,
            qps=qps,
            latency_ms=latency_ms,
        )

    def _snap_range_ratio(self, ratio: float) -> float:
        """
        Snap a parsed ratio to the nearest configured bucket.

        If the closest bucket is farther than tolerance, we keep the raw ratio.
        """
        best = self._range_ratios[0]
        best_diff = abs(ratio - best)
        for candidate in self._range_ratios[1:]:
            diff = abs(ratio - candidate)
            if diff < best_diff:
                best = candidate
                best_diff = diff
        if best_diff <= self._tol:
            return best
        return ratio


def build_recall_qps_points(
    grouped: Mapping[float, Sequence[BenchmarkRecord]],
) -> Dict[float, List[Tuple[float, float]]]:
    """
    Convert grouped records into (recall, qps) pairs per range ratio.

    This helper is useful for plotting or exporting to CSV.
    """
    out: Dict[float, List[Tuple[float, float]]] = {}
    for ratio, recs in grouped.items():
        pts = [(r.recall, r.qps) for r in recs]
        out[ratio] = pts
    return out


