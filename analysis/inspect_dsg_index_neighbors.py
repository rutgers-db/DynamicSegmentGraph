"""
inspect_dsg_index_neighbors.py
Author: Zhencan Peng, 2025/11/30

This script inspects a saved Dynamic Segment Graph (DSG) binary index (format "DSGIDX3")
and reports neighbor statistics that help explain surprising query behavior, e.g.:

- For very small ranges (e.g. 1% of N), `rangeSearch()` can intentionally skip the
  envelope checks (ll/lu/rl/ru), making the effective filter only "neighbor_id in [L, R]".
  If neighbors are heavily concentrated near the node label in sorted-by-timestamp datasets,
  many neighbors will lie inside a 1% window, causing high distance-evaluation counts.

This tool reads only the metadata arrays fully (row offsets, degrees, labels) and
reads adjacency slices on demand for sampled rows; it does NOT load all edges.

Usage:
  python3 analysis/inspect_dsg_index_neighbors.py \
    --index_path /path/to/*.index \
    --data_size 1000000 \
    --sample_rows 2000 \
    --seed 42

Optional:
  --dump_label 123456  (print neighbors + envelope stats for a specific label)

Outputs:
  - Degree distribution (mean/p50/p90/p99/max)
  - Neighbor label-offset distribution (abs(neighbor - center)) quantiles
  - For a given window ratio (default 1% and 2%), estimate how many neighbors are
    admitted at typical seed positions (left, quarter, mid, 3/4) under:
      * id-only filter (neighbors in [L,R])
      * id+envelope filter (also passing ll/lu/rl/ru)

Complexity:
  - Time: O(S * (deg + log deg)) where S is sampled rows, deg is average degree.
  - Space: O(num_rows) for metadata + O(max_deg) for one-row buffers.
"""

from __future__ import annotations

import argparse
import bisect
import os
import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


MAGIC = b"DSGIDX3\x00"
VERSION = 3


@dataclass(frozen=True)
class IndexLayout:
    data_size: int
    num_rows: int
    size_t_bytes: int
    offset_row_to_label: int
    offset_node_degrees: int
    offset_row_offset: int
    offset_num_edges_total: int
    offset_neighbors: int
    offset_ll: int
    offset_lu: int
    offset_rl: int
    offset_ru: int
    num_edges_total: int


def _read_exact(f, nbytes: int) -> bytes:
    buf = f.read(nbytes)
    if len(buf) != nbytes:
        raise RuntimeError(f"Unexpected EOF: need {nbytes} bytes, got {len(buf)}")
    return buf


def _u32_list_from_file(f, count: int) -> List[int]:
    # Read as bytes then unpack. This is OK for ~1e6 entries.
    raw = _read_exact(f, 4 * count)
    # Little-endian u32
    return list(struct.unpack("<" + "I" * count, raw))


def _u64_list_from_file(f, count: int) -> List[int]:
    raw = _read_exact(f, 8 * count)
    return list(struct.unpack("<" + "Q" * count, raw))


def _node_degree_list_from_file(f, count: int) -> List[Tuple[int, int]]:
    # NodeDegree is (uint16 fwd, uint16 rev), 4 bytes per row.
    raw = _read_exact(f, 4 * count)
    vals = struct.unpack("<" + "HH" * count, raw)
    out: List[Tuple[int, int]] = []
    out.reserve = None  # type: ignore[attr-defined]
    # Convert flat tuple to pairs.
    for i in range(0, len(vals), 2):
        out.append((int(vals[i]), int(vals[i + 1])))
    return out


def parse_index_layout(index_path: str, size_t_bytes: int = 8) -> Tuple[IndexLayout, List[int], List[Tuple[int, int]], List[int]]:
    if size_t_bytes not in (4, 8):
        raise ValueError("--size_t_bytes must be 4 or 8")

    with open(index_path, "rb") as f:
        magic = _read_exact(f, 8)
        if magic != MAGIC:
            raise RuntimeError(f"Bad magic header: got {magic!r}, expect {MAGIC!r}")

        version = struct.unpack("<I", _read_exact(f, 4))[0]
        if version != VERSION:
            raise RuntimeError(f"Unsupported version: got {version}, expect {VERSION}")

        data_size = struct.unpack("<Q", _read_exact(f, 8))[0]
        num_rows = struct.unpack("<Q", _read_exact(f, 8))[0]

        if data_size > 2**31 or num_rows > 2**31:
            raise RuntimeError("Index metadata too large for this inspector.")

        data_size_i = int(data_size)
        num_rows_i = int(num_rows)

        offset_row_to_label = f.tell()
        row_to_label = _u32_list_from_file(f, num_rows_i)

        offset_node_degrees = f.tell()
        node_degrees = _node_degree_list_from_file(f, num_rows_i)

        offset_row_offset = f.tell()
        if size_t_bytes == 8:
            row_offset = _u64_list_from_file(f, num_rows_i + 1)
        else:
            row_offset = _u32_list_from_file(f, num_rows_i + 1)

        offset_num_edges_total = f.tell()
        num_edges_total = struct.unpack("<Q", _read_exact(f, 8))[0]
        num_edges_total_i = int(num_edges_total)

        offset_neighbors = f.tell()
        neighbors_bytes = 4 * num_edges_total_i
        offset_ll = offset_neighbors + neighbors_bytes
        offset_lu = offset_ll + neighbors_bytes
        offset_rl = offset_lu + neighbors_bytes
        offset_ru = offset_rl + neighbors_bytes

        layout = IndexLayout(
            data_size=data_size_i,
            num_rows=num_rows_i,
            size_t_bytes=size_t_bytes,
            offset_row_to_label=offset_row_to_label,
            offset_node_degrees=offset_node_degrees,
            offset_row_offset=offset_row_offset,
            offset_num_edges_total=offset_num_edges_total,
            offset_neighbors=offset_neighbors,
            offset_ll=offset_ll,
            offset_lu=offset_lu,
            offset_rl=offset_rl,
            offset_ru=offset_ru,
            num_edges_total=num_edges_total_i,
        )
        return layout, row_to_label, node_degrees, [int(x) for x in row_offset]


def _read_u32_slice(path: str, base_offset: int, start: int, end: int) -> List[int]:
    if end <= start:
        return []
    n = end - start
    with open(path, "rb") as f:
        f.seek(base_offset + 4 * start)
        raw = _read_exact(f, 4 * n)
    return list(struct.unpack("<" + "I" * n, raw))


def _clamp_window(left: int, window: int, data_size: int) -> Tuple[int, int]:
    # Clamp so that [L, R] is valid and has exactly `window` length.
    if window <= 0:
        raise ValueError("window must be positive")
    if window >= data_size:
        return 0, data_size - 1
    left = max(0, min(left, data_size - window))
    right = left + window - 1
    return left, right


def _seed_windows_for_center(center: int, window: int, data_size: int) -> List[Tuple[str, int, int]]:
    # Mirrors the four anchors used in rangeSearch(): left, mid, quarter, 3/4.
    # For a *given center label*, we compute the window that would place this center at
    # that anchor position, then clamp to [0, N-window].
    if window <= 1:
        return [("left", center, center)]
    # anchor positions in [0, 1]
    anchors = [
        ("left", 0.0),
        ("quarter", 0.25),
        ("mid", 0.5),
        ("three_quarter", 0.75),
    ]
    out: List[Tuple[str, int, int]] = []
    span = window - 1
    for name, frac in anchors:
        delta = int(round(frac * span))
        l = center - delta
        l, r = _clamp_window(l, window, data_size)
        out.append((name, l, r))
    return out


def _quantiles(sorted_vals: Sequence[int], ps: Sequence[float]) -> List[int]:
    if not sorted_vals:
        return [0 for _ in ps]
    n = len(sorted_vals)
    out: List[int] = []
    for p in ps:
        # nearest-rank style
        idx = int(round(p * (n - 1)))
        idx = max(0, min(idx, n - 1))
        out.append(int(sorted_vals[idx]))
    return out


def _percentiles_str(sorted_vals: Sequence[int]) -> str:
    qs = _quantiles(sorted_vals, [0.5, 0.9, 0.99])
    return f"p50={qs[0]}, p90={qs[1]}, p99={qs[2]}"


def inspect(
    index_path: str,
    size_t_bytes: int,
    sample_rows: int,
    seed: int,
    dump_label: Optional[int],
    data_size_override: Optional[int],
) -> None:
    layout, row_to_label, node_degrees, row_offset = parse_index_layout(
        index_path=index_path,
        size_t_bytes=size_t_bytes,
    )
    data_size = layout.data_size
    if data_size_override is not None:
        data_size = data_size_override

    # Basic degree stats from metadata (do not touch adjacency arrays).
    degrees: List[int] = []
    for fwd, rev in node_degrees:
        degrees.append(int(fwd) + int(rev))
    degrees_sorted = sorted(degrees)
    deg_mean = sum(degrees) / max(1, len(degrees))
    deg_max = max(degrees) if degrees else 0
    deg_q = _quantiles(degrees_sorted, [0.5, 0.9, 0.99])

    print("== DSGIDX3 index metadata ==")
    print(f"index_path: {index_path}")
    print(f"data_size: {layout.data_size} (override used: {data_size})")
    print(f"num_rows: {layout.num_rows}")
    print(f"num_edges_total: {layout.num_edges_total}")
    print(f"size_t_bytes (assumed): {size_t_bytes}")
    print("")
    print("== Degree distribution (from node_degrees_) ==")
    print(f"mean={deg_mean:.3f}, p50={deg_q[0]}, p90={deg_q[1]}, p99={deg_q[2]}, max={deg_max}")
    print("")

    # Decide sampled row ids (uniform in [0, num_rows)).
    # Avoid importing random for reproducibility with simple LCG.
    nrows = layout.num_rows
    if sample_rows <= 0:
        sample_rows = 1
    sample_rows = min(sample_rows, nrows)
    x = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    sampled: List[int] = []
    seen = set()
    while len(sampled) < sample_rows:
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        rid = int(x % nrows)
        if rid in seen:
            continue
        seen.add(rid)
        sampled.append(rid)

    # Inspect adjacency slices for sampled rows.
    # Metrics:
    # - abs offset distribution
    # - "admitted neighbor count" for 1% and 2% windows, for anchor positions.
    offset_vals: List[int] = []
    thresholds = [100, 500, 1000, 2500, 5000, 10000, 20000]
    offset_threshold_hits = {t: 0 for t in thresholds}
    offset_total = 0

    # Window sizes
    w1 = max(1, int(round(data_size * 0.01)))
    w2 = max(1, int(round(data_size * 0.02)))

    # Aggregated admitted counts per anchor for w1 and w2
    anchors = ["left", "quarter", "mid", "three_quarter"]
    w1_id_only = {a: 0 for a in anchors}
    w2_id_only = {a: 0 for a in anchors}
    w2_id_plus_env = {a: 0 for a in anchors}
    sampled_rows_used = 0

    # Track extreme rows (largest admitted counts) for debugging.
    top_w1_mid: List[Tuple[int, int, int]] = []  # (count, row, center_label)

    for row in sampled:
        start = row_offset[row]
        end = row_offset[row + 1]
        if end <= start:
            continue
        deg = end - start
        if deg <= 0:
            continue

        center = int(row_to_label[row])

        nbrs = _read_u32_slice(index_path, layout.offset_neighbors, start, end)
        if not nbrs:
            continue

        # Optional: sanity that forward is sorted (assumed by rangeSearch()).
        # We avoid raising in loops; just skip if corrupted.
        is_sorted = True
        for i in range(1, len(nbrs)):
            if nbrs[i] < nbrs[i - 1]:
                is_sorted = False
                break
        if not is_sorted:
            continue

        # Offset distribution.
        local_offsets = [abs(int(v) - center) for v in nbrs]
        offset_vals.extend(local_offsets)
        offset_total += len(local_offsets)
        for t in thresholds:
            # Count offsets <= t
            hit = 0
            for d in local_offsets:
                if d <= t:
                    hit += 1
            offset_threshold_hits[t] += hit

        # For w2 envelope filter we need envelopes for this row.
        ll = _read_u32_slice(index_path, layout.offset_ll, start, end)
        lu = _read_u32_slice(index_path, layout.offset_lu, start, end)
        rl = _read_u32_slice(index_path, layout.offset_rl, start, end)
        ru = _read_u32_slice(index_path, layout.offset_ru, start, end)

        # Anchor windows for center.
        sampled_rows_used += 1
        for name, L, R in _seed_windows_for_center(center, w1, data_size):
            if name not in w1_id_only:
                continue
            # Count neighbors in [L,R] using binary search.
            i0 = bisect.bisect_left(nbrs, L)
            i1 = bisect.bisect_right(nbrs, R)
            w1_id_only[name] += (i1 - i0)
            if name == "mid":
                top_w1_mid.append((i1 - i0, row, center))

        for name, L, R in _seed_windows_for_center(center, w2, data_size):
            if name not in w2_id_only:
                continue
            i0 = bisect.bisect_left(nbrs, L)
            i1 = bisect.bisect_right(nbrs, R)
            w2_id_only[name] += (i1 - i0)

            # Envelope check among the in-range neighbors
            pass_env = 0
            for i in range(i0, i1):
                # Need L in [ll,lu] and R in [rl,ru]
                if ll[i] <= L <= lu[i] and rl[i] <= R <= ru[i]:
                    pass_env += 1
            w2_id_plus_env[name] += pass_env

    print("== Neighbor |id-center| distribution (sampled) ==")
    if offset_vals:
        offset_vals_sorted = sorted(offset_vals)
        q = _quantiles(offset_vals_sorted, [0.5, 0.9, 0.99])
        print(f"count={len(offset_vals_sorted)} | { _percentiles_str(offset_vals_sorted) }")
        print(f"min={offset_vals_sorted[0]}, max={offset_vals_sorted[-1]}")
        print("fraction of neighbors within thresholds:")
        for t in thresholds:
            frac = offset_threshold_hits[t] / max(1, offset_total)
            print(f"  <= {t:5d}: {frac*100:6.2f}%")
    else:
        print("No offsets collected (sampling may have skipped all rows).")
    print("")

    if sampled_rows_used > 0:
        print("== Estimated admitted neighbors at seed positions ==")
        print(f"window_1pct: {w1}  (id-only; envelope is skipped in current rangeSearch() impl)")
        for a in anchors:
            print(f"  {a:13s}: avg_in_window={w1_id_only[a] / sampled_rows_used:.2f}")
        print("")
        print(f"window_2pct: {w2}  (id-only vs id+envelope)")
        for a in anchors:
            avg_id = w2_id_only[a] / sampled_rows_used
            avg_env = w2_id_plus_env[a] / sampled_rows_used
            print(f"  {a:13s}: avg_id_only={avg_id:.2f}  avg_id_plus_env={avg_env:.2f}")
        print("")

        # Show a few extreme rows for w1 mid-window.
        top_w1_mid.sort(reverse=True)
        print("== Top rows by (1% window, mid anchor) admitted neighbor count ==")
        for cnt, row, center in top_w1_mid[:10]:
            print(f"  row={row} label={center} admitted={cnt}")
        print("")

    if dump_label is not None:
        if dump_label < 0 or dump_label >= layout.data_size:
            print(f"dump_label out of range: {dump_label}")
            return
        # Find row: row_to_label is dense in typical static build; linear scan is OK once.
        # (If you want faster, build an inverse mapping.)
        row_id = -1
        for i, lbl in enumerate(row_to_label):
            if int(lbl) == dump_label:
                row_id = i
                break
        if row_id < 0:
            print(f"Label {dump_label} not found in row_to_label_.")
            return

        start = row_offset[row_id]
        end = row_offset[row_id + 1]
        nbrs = _read_u32_slice(index_path, layout.offset_neighbors, start, end)
        ll = _read_u32_slice(index_path, layout.offset_ll, start, end)
        lu = _read_u32_slice(index_path, layout.offset_lu, start, end)
        rl = _read_u32_slice(index_path, layout.offset_rl, start, end)
        ru = _read_u32_slice(index_path, layout.offset_ru, start, end)
        offs = [abs(int(v) - dump_label) for v in nbrs]
        print("== Dump label adjacency ==")
        print(f"label={dump_label}, row={row_id}, degree={len(nbrs)}")
        if offs:
            print(f"|nbr-label|: {_percentiles_str(sorted(offs))}")
        # Print first few and a few around median offset.
        show = min(40, len(nbrs))
        print(f"first {show} edges (nbr, |d|, [ll,lu]x[rl,ru]):")
        for i in range(show):
            print(f"  {i:3d}: nbr={nbrs[i]:7d}  |d|={offs[i]:6d}  "
                  f"LL={ll[i]:7d} LU={lu[i]:7d}  RL={rl[i]:7d} RU={ru[i]:7d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True, type=str)
    ap.add_argument("--size_t_bytes", type=int, default=8, help="Assumed size_t bytes in the index file (4 or 8).")
    ap.add_argument("--sample_rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_label", type=int, default=None)
    ap.add_argument("--data_size", type=int, default=None, help="Override data_size for window computations.")
    args = ap.parse_args()

    if not os.path.exists(args.index_path):
        raise RuntimeError(f"index_path does not exist: {args.index_path}")

    inspect(
        index_path=args.index_path,
        size_t_bytes=args.size_t_bytes,
        sample_rows=args.sample_rows,
        seed=args.seed,
        dump_label=args.dump_label,
        data_size_override=args.data_size,
    )


if __name__ == "__main__":
    main()

