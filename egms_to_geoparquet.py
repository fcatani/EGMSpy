#!/usr/bin/env python3
"""
EGMS CSV to GeoParquet Converter  (v3)
MIT License

Copyright (c) 2026 Filippo Catani and MISS-Lab at UNIPD
========================================
Converts EGMS InSAR CSV files (Copernicus Land Monitoring Service) into
compact, split GeoParquet format.

Architecture (v3)
-----------------
  ONE pair of Parquet files is produced PER input CSV:
    <name>_meta.parquet  —  25 metadata cols + WGS84 geometry (GeoParquet 1.0)
    <name>_ts.parquet    —  pid + time series displacement columns (float32)

  All pairs live in the same output folder and are queried together by the
  bridge/viewer via DuckDB wildcards:
    read_parquet('processed_data/*_meta.parquet', union_by_name=True)
    read_parquet('processed_data/*_ts.parquet',   union_by_name=True)

  union_by_name=True handles different date columns across files (different
  tracks, time windows) — DuckDB fills missing dates with NULL.

Key features
------------
  • Skip logic      : already-processed files are skipped (safe to re-run)
  • Orbit column    : 25th metadata column derived from track_angle % 360
  • Compact dtypes  : float32 for TS, typed metadata → ~70% smaller than CSV
  • GeoParquet 1.0  : QGIS-compatible, WGS84 EPSG:4326, covering bbox index
  • Gap metrics     : max_gap_days + n_acquisitions added to metadata
  • Rich console    : detailed progress output per file

Orbit detection rule (Sentinel-1 / EGMS)
-----------------------------------------
  90 < (track_angle % 360) < 270  →  Descending ('D')
  otherwise                        →  Ascending  ('A')
  Handles negative angles, angles > 360, and all EGMS processing versions.

Usage
-----
  # Basic (output: <input-dir>/processed_data/)
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw

  # Custom output folder
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --out-dir C:/data/processed

  # With quality pre-filters
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw \\
      --min-coherence 0.7 --abs-velocity-limit 200

  # Better compression (smaller files, slightly slower)
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --compression zstd

  # Low RAM mode for very large files
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --chunksize 50000

Requirements
------------
  conda install -c conda-forge pandas pyarrow geopandas shapely tqdm
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from tqdm import tqdm

import re as _re

# ── Date column helpers ────────────────────────────────────────────────────────
# EGMS raw CSV files name date columns as D20190401 (D prefix + YYYYMMDD).
# We strip the D prefix at conversion time so all parquet files store bare
# YYYYMMDD column names (e.g. 20190401). This makes DuckDB, pandas, and
# all downstream tools work without special-casing the prefix.

_EGMS_DATE_PAT = _re.compile(r'^D?(\d{8})$')

def _is_egms_date_col(c: str) -> bool:
    """True for D20190401 or 20190401 style column names."""
    return bool(_EGMS_DATE_PAT.match(str(c).strip()))

def _bare_date(c: str) -> str:
    """Strip D prefix → bare YYYYMMDD string. '20190401' stays unchanged."""
    m = _EGMS_DATE_PAT.match(str(c).strip())
    return m.group(1) if m else str(c)

def _fmt_date(c: str) -> str:
    """Return YYYY-MM-DD display form."""
    s = _bare_date(c)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"



# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Exact column names from EGMS technical specification (24 metadata columns)
METADATA_COLS = [
    "pid", "mp_type",
    "latitude", "longitude", "easting", "northing",
    "height", "height_wgs84",
    "line", "pixel",
    "rmse", "temporal_coherence", "amplitude_dispersion",
    "incidence_angle", "track_angle",
    "los_east", "los_north", "los_up",
    "mean_velocity", "mean_velocity_std",
    "acceleration", "acceleration_std",
    "seasonality", "seasonality_std",
    # 'orbit' → 25th column, derived at conversion time
]

# Compact dtype map — saves ~40% on metadata file size
META_DTYPES = {
    "pid":                  "str",
    "mp_type":              "Int8",
    "latitude":             "float64",    # full precision for geometry
    "longitude":            "float64",
    "easting":              "float32",
    "northing":             "float32",
    "height":               "float32",
    "height_wgs84":         "float32",
    "line":                 "int32",
    "pixel":                "int32",
    "rmse":                 "float32",
    "temporal_coherence":   "float32",
    "amplitude_dispersion": "float32",
    "incidence_angle":      "float32",
    "track_angle":          "float32",
    "los_east":             "float32",
    "los_north":            "float32",
    "los_up":               "float32",
    "mean_velocity":        "float32",
    "mean_velocity_std":    "float32",
    "acceleration":         "float32",
    "acceleration_std":     "float32",
    "seasonality":          "float32",
    "seasonality_std":      "float32",
}

# Console formatting
SEP  = "═" * 65
SEP2 = "─" * 65


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_orbit(track_angle_series: pd.Series) -> pd.Series:
    """
    Derive Sentinel-1 orbit from track_angle using modulo-360 normalisation.

      90 < (track_angle % 360) < 270  →  'D' (Descending, heading southward)
      otherwise                        →  'A' (Ascending,  heading northward)
      NaN                              →  'U' (Unknown)
    """
    angles    = pd.to_numeric(track_angle_series, errors="coerce")
    normalised = angles % 360
    values = np.where(
        angles.isna(), "U",
        np.where((normalised > 90) & (normalised < 270), "D", "A")
    )
    return pd.Series(
        pd.Categorical(values, categories=["A", "D", "U"]),
        index=track_angle_series.index,
        name="orbit",
    )


def compute_gap_metrics(ts_cols: list[str]) -> tuple[int, int]:
    """
    Given a list of date column names (DYYYYMMDD or YYYYMMDD), return:
      (max_gap_days, n_acquisitions)
    max_gap_days: longest gap in days between consecutive acquisitions.
    """
    n = len(ts_cols)
    if n == 0:
        return 0, 0
    try:
        bare  = [_bare_date(c) for c in ts_cols]
        dates = pd.to_datetime(bare, format="%Y%m%d")
        gaps  = dates.to_series().diff().dt.days.dropna()
        max_gap = int(gaps.max()) if len(gaps) > 0 else 0
    except Exception:
        max_gap = 0
    return max_gap, n


def count_lines_fast(path: Path) -> int:
    """Fast byte-level line counter. Returns number of data rows (excl. header)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1_000_000:.1f} MB"


def fmt_size(n_bytes: int) -> str:
    return f"{n_bytes / 1_000_000:.1f} MB"


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FILE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def make_bbox_subdir_name(west, south, east, north):
    """
    Build a compact, filesystem-safe subfolder name from a WGS84 bbox.
    Negative values use suffix W/S to avoid minus signs in folder names.
    Example:  bbox_N45.2000N46.1000_E10.8000E12.3000
              bbox_N63.0000N66.0000_W25.0000W13.0000  (Iceland)
    """
    def _c(v, pos_sfx, neg_sfx):
        return f"{abs(v):.4f}{pos_sfx if v >= 0 else neg_sfx}"
    return (
        f"bbox_{_c(south,'N','S')}{_c(north,'N','S')}"
        f"_{_c(west,'E','W')}{_c(east,'E','W')}"
    )


def convert_single_file(
    csv_path:            Path,
    out_dir:             Path,
    chunksize:           int   = 200_000,
    compression:         str   = "snappy",
    min_coherence:       float | None = None,
    abs_velocity_limit:  float | None = None,
    bbox:                tuple | None = None,
) -> dict:
    """
    Convert one EGMS CSV file to a pair of Parquet files.

    bbox: optional (west, south, east, north) in WGS84 decimal degrees.
          Points outside this rectangle are discarded at read time,
          before any dtype casting — zero RAM overhead for excluded rows.

    Returns a result dict with statistics for the batch summary.
    Raises RuntimeError if no time series columns are detected.
    """
    t0       = time.time()
    stem     = csv_path.stem
    meta_out = out_dir / f"{stem}_meta.parquet"
    ts_out   = out_dir / f"{stem}_ts.parquet"

    # ── Column layout scan ────────────────────────────────────────────────
    header         = pd.read_csv(csv_path, nrows=0).columns.tolist()
    meta_cols_here = [c for c in METADATA_COLS if c in header]
    ts_cols_raw    = [c for c in header if c not in METADATA_COLS]
    # Build rename map: D20190401 → 20190401 (strip D prefix for clean storage)
    ts_rename      = {c: _bare_date(c) for c in ts_cols_raw if _is_egms_date_col(c)}
    ts_cols_raw    = [c for c in ts_cols_raw if _is_egms_date_col(c)]
    ts_cols        = [ts_rename[c] for c in ts_cols_raw]   # bare YYYYMMDD names
    max_gap, n_acq = compute_gap_metrics(ts_cols)

    print(f"\n  {SEP2}")
    print(f"  File   : {csv_path.name}")
    print(f"  Meta columns   : {len(meta_cols_here):>4d}")
    print(f"  TS   columns   : {len(ts_cols):>4d}  "
          f"({ts_cols[0] if ts_cols else '?'} → {ts_cols[-1] if ts_cols else '?'})")
    print(f"  Max gap        : {max_gap:>4d} days   "
          f"Acquisitions: {n_acq}")
    if min_coherence is not None:
        print(f"  Filter         : coherence >= {min_coherence}")
    if abs_velocity_limit is not None:
        print(f"  Filter         : |velocity| <= {abs_velocity_limit} mm/yr")
    if bbox is not None:
        _bw, _bs, _be, _bn = bbox
        print(f"  Bbox filter    : W{_bw:.4f} S{_bs:.4f} E{_be:.4f} N{_bn:.4f}")

    if not ts_cols:
        raise RuntimeError(
            f"No time series columns found in {csv_path.name}.\n"
            "Expected date columns in YYYYMMDD format. "
            "Check that METADATA_COLS covers all non-TS column names exactly."
        )

    n_total     = count_lines_fast(csv_path)
    n_filtered  = 0
    meta_chunks = []
    ts_chunks   = []

    # ── Chunked read ──────────────────────────────────────────────────────
    with tqdm(
        total=n_total,
        unit="rows",
        desc="  Reading ",
        leave=False,
        bar_format="{desc}{percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:

        for chunk in pd.read_csv(
            csv_path,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        ):
            n_chunk = len(chunk)

            # ── Cast metadata dtypes ──────────────────────────────────────
            meta_chunk = chunk[meta_cols_here].copy()
            for col, dtype in META_DTYPES.items():
                if col in meta_chunk.columns and col != "pid":
                    meta_chunk[col] = (
                        pd.to_numeric(meta_chunk[col], errors="coerce")
                        .astype(dtype)
                    )

            # ── Orbit (25th column) ───────────────────────────────────────
            meta_chunk["orbit"] = detect_orbit(meta_chunk["track_angle"])

            # ── Gap metrics (same value for every row in this file) ───────
            meta_chunk["max_gap_days"]   = np.int16(max_gap)
            meta_chunk["n_acquisitions"] = np.int16(n_acq)

            # ── Spatial bbox filter (row-level, before quality filters) ──
            if bbox is not None:
                _bw, _bs, _be, _bn = bbox
                _lat = meta_chunk["latitude"]
                _lon = meta_chunk["longitude"]
                _in_box = (
                    (_lat >= _bs) & (_lat <= _bn) &
                    (_lon >= _bw) & (_lon <= _be)
                )
                meta_chunk = meta_chunk[_in_box].reset_index(drop=True)
                chunk      = chunk[_in_box].reset_index(drop=True)

            # ── Quality filters ───────────────────────────────────────────
            keep = pd.Series(True, index=meta_chunk.index)
            if min_coherence is not None:
                keep &= meta_chunk["temporal_coherence"] >= min_coherence
            if abs_velocity_limit is not None:
                keep &= meta_chunk["mean_velocity"].abs() <= abs_velocity_limit
            n_filtered += int((~keep).sum())

            meta_chunk = meta_chunk[keep].reset_index(drop=True)

            # ── Time series (float32) ─────────────────────────────────────
            # Select raw columns (D20190401 etc.), rename to bare YYYYMMDD
            ts_chunk = chunk[["pid"]].copy()
            ts_data  = (
                chunk[ts_cols_raw]
                .rename(columns=ts_rename)
                .apply(pd.to_numeric, errors="coerce")
                .astype("float32")
            )
            ts_chunk = pd.concat([ts_chunk, ts_data], axis=1)
            ts_chunk = ts_chunk[keep].reset_index(drop=True)

            if len(meta_chunk) > 0:
                meta_chunks.append(meta_chunk)
                ts_chunks.append(ts_chunk)

            pbar.update(n_chunk)

    # ── Concatenate ───────────────────────────────────────────────────────
    meta_df = pd.concat(meta_chunks, ignore_index=True)
    ts_df   = pd.concat(ts_chunks,   ignore_index=True)
    n_pts   = len(meta_df)

    # ── Orbit distribution ────────────────────────────────────────────────
    orbit_counts = meta_df["orbit"].value_counts()

    # ── Build GeoDataFrame ────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(
        meta_df,
        geometry=gpd.points_from_xy(
            meta_df["longitude"].astype("float64"),
            meta_df["latitude"].astype("float64"),
        ),
        crs="EPSG:4326",
    )

    # ── Write GeoParquet (metadata) ───────────────────────────────────────
    gdf.to_parquet(
        meta_out,
        compression=compression,
        index=False,
        write_covering_bbox=True,
        schema_version="1.0.0",
    )
    # Embed searchable file-level metadata
    _inject_parquet_metadata(meta_out, {
        "egms_source_file":         csv_path.name,
        "egms_ts_start":            ts_cols[0],
        "egms_ts_end":              ts_cols[-1],
        "egms_ts_n_dates":          str(n_acq),
        "egms_ts_max_gap_days":     str(max_gap),
        "egms_n_points":            str(n_pts),
        "egms_min_coherence":       str(min_coherence),
        "egms_abs_velocity_limit":  str(abs_velocity_limit),
        "egms_orbit_encoding":      "A=Ascending D=Descending U=Unknown",
        "egms_ts_dates":            json.dumps(ts_cols),
    })

    # ── Write TS Parquet ──────────────────────────────────────────────────
    ts_table = pa.Table.from_pandas(ts_df, preserve_index=False)
    ts_table = ts_table.replace_schema_metadata({
        b"egms_ts_dates":  json.dumps(ts_cols).encode(),
        b"egms_n_points":  str(n_pts).encode(),
    })
    pq.write_table(
        ts_table,
        ts_out,
        compression=compression,
        row_group_size=50_000,
    )

    elapsed = time.time() - t0

    # ── Per-file result summary ───────────────────────────────────────────
    in_mb   = csv_path.stat().st_size / 1_000_000
    out_mb  = (meta_out.stat().st_size + ts_out.stat().st_size) / 1_000_000
    saving  = 100 * (1 - out_mb / in_mb) if in_mb > 0 else 0

    print(f"  Points written : {n_pts:>10,}  (filtered: {n_filtered:,})")
    for orb, cnt in orbit_counts.items():
        label = {"A": "Ascending", "D": "Descending", "U": "Unknown"}.get(str(orb), str(orb))
        print(f"  Orbit {label:<12}: {cnt:>10,}")
    print(f"  Input size     : {in_mb:>9.1f} MB")
    print(f"  Output size    : {out_mb:>9.1f} MB  (saved {saving:.0f}%)")
    print(f"  ✓ Done         : {elapsed:>9.1f} s")

    return {
        "csv_path":      csv_path,
        "meta_out":      meta_out,
        "ts_out":        ts_out,
        "n_points":      n_pts,
        "n_filtered":    n_filtered,
        "n_ts_cols":     n_acq,
        "max_gap_days":  max_gap,
        "orbit_counts":  dict(orbit_counts),
        "in_mb":         in_mb,
        "out_mb":        out_mb,
        "elapsed_sec":   elapsed,
        "skipped":       False,
    }


def _inject_parquet_metadata(path: Path, new_meta: dict):
    """Append key/value pairs to an existing Parquet file's schema metadata.

    Uses pyarrow directly but preserves ALL existing metadata (including the
    GeoParquet 'geo' key and any covering bbox column written by GeoPandas).
    The file is rewritten in-place atomically via a temp file so a crash
    mid-write never leaves a corrupt output.
    """
    import tempfile, os
    table    = pq.read_table(path)
    existing = table.schema.metadata or {}
    merged   = {
        **existing,
        **{
            (k.encode() if isinstance(k, str) else k):
            (v.encode() if isinstance(v, str) else v)
            for k, v in new_meta.items()
        },
    }
    table = table.replace_schema_metadata(merged)
    # Write to a temp file in the same directory, then atomically replace
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(tmp_fd)
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)   # atomic on all major OSes
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def convert_folder(
    input_dir:           Path,
    out_dir:             Path,
    chunksize:           int   = 200_000,
    compression:         str   = "snappy",
    min_coherence:       float | None = None,
    abs_velocity_limit:  float | None = None,
    force:               bool  = False,
    bbox:                tuple | None = None,   # (west, south, east, north) WGS84
) -> list[dict]:
    """
    Process all CSV/TXT files in input_dir.
    Each file produces its own <name>_meta.parquet + <name>_ts.parquet.
    Already-processed files are skipped unless --force is set.
    """
    t_batch   = time.time()
    input_dir = Path(input_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV/TXT files directly, or inside ZIP archives
    csv_files = sorted(
        list(input_dir.glob("*.csv")) + list(input_dir.glob("*.txt"))
    )
    zip_files = sorted(input_dir.glob("*.zip"))

    if not csv_files and not zip_files:
        raise FileNotFoundError(
            f"No .csv, .txt, or .zip files found in {input_dir}"
        )

    # If only ZIPs found, extract them first into input_dir
    if not csv_files and zip_files:
        import zipfile as _zipfile
        print(f"  Found {len(zip_files)} ZIP files — extracting CSVs ...")
        for zp in zip_files:
            csv_stem = zp.stem
            csv_out  = input_dir / f"{csv_stem}.csv"
            if csv_out.exists():
                continue   # already extracted
            try:
                with _zipfile.ZipFile(zp) as z:
                    names = [n for n in z.namelist() if n.endswith(".csv")]
                    if names:
                        z.extract(names[0], path=input_dir)
                        # rename to match ZIP stem if needed
                        extracted = input_dir / names[0]
                        if extracted != csv_out and extracted.exists():
                            extracted.rename(csv_out)
                        print(f"    ✓ {zp.name}")
            except Exception as ex:
                print(f"    ✗ {zp.name}: {ex}")
        csv_files = sorted(
            list(input_dir.glob("*.csv")) + list(input_dir.glob("*.txt"))
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No CSVs could be extracted from ZIPs in {input_dir}"
            )

    # ── Batch header ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EGMS CSV → GeoParquet Converter  (v3)")
    print(f"{SEP}")
    print(f"  Input  folder  : {input_dir}")
    print(f"  Output folder  : {out_dir}")
    print(f"  Files found    : {len(csv_files)}")
    print(f"  Compression    : {compression}")
    if bbox is not None:
        _bw, _bs, _be, _bn = bbox
        print(f"  Bbox filter    : W{_bw:.4f} S{_bs:.4f} E{_be:.4f} N{_bn:.4f}")
    if min_coherence is not None:
        print(f"  Filter         : temporal_coherence >= {min_coherence}")
    if abs_velocity_limit is not None:
        print(f"  Filter         : |mean_velocity| <= {abs_velocity_limit} mm/yr")
    print(f"  Skip existing  : {'No (--force)' if force else 'Yes'}")
    print(f"{SEP}")

    # ── File inventory ────────────────────────────────────────────────────
    print(f"\n  {'#':>3}  {'Filename':<50}  {'Size':>8}  {'Status'}")
    print(f"  {'─'*3}  {'─'*50}  {'─'*8}  {'─'*12}")
    for i, p in enumerate(csv_files, 1):
        meta_exists = (out_dir / f"{p.stem}_meta.parquet").exists()
        status = "SKIP (done)" if (meta_exists and not force) else "PENDING"
        print(f"  {i:>3}  {p.name:<50}  {fmt_size(p.stat().st_size):>8}  {status}")
    print()

    # ── Process each file ─────────────────────────────────────────────────
    results = []
    for i, csv_path in enumerate(csv_files, 1):
        meta_out = out_dir / f"{csv_path.stem}_meta.parquet"
        ts_out   = out_dir / f"{csv_path.stem}_ts.parquet"

        print(f"\n  [{i}/{len(csv_files)}]", end="")

        # Skip logic
        if meta_out.exists() and ts_out.exists() and not force:
            print(f" SKIPPING {csv_path.name}  (already processed)")
            results.append({
                "csv_path":    csv_path,
                "meta_out":    meta_out,
                "ts_out":      ts_out,
                "skipped":     True,
                "n_points":    0,
                "in_mb":       csv_path.stat().st_size / 1_000_000,
                "out_mb":      (meta_out.stat().st_size + ts_out.stat().st_size) / 1_000_000,
                "elapsed_sec": 0,
            })
            continue

        try:
            result = convert_single_file(
                csv_path=csv_path,
                out_dir=out_dir,
                chunksize=chunksize,
                compression=compression,
                min_coherence=min_coherence,
                abs_velocity_limit=abs_velocity_limit,
                bbox=bbox,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ ERROR processing {csv_path.name}:")
            print(f"    {e}")
            results.append({
                "csv_path":    csv_path,
                "skipped":     False,
                "error":       str(e),
                "n_points":    0,
                "in_mb":       csv_path.stat().st_size / 1_000_000,
                "out_mb":      0,
                "elapsed_sec": 0,
            })

    # ── Batch summary ─────────────────────────────────────────────────────
    elapsed_total = time.time() - t_batch
    _print_batch_summary(results, elapsed_total, out_dir)

    return results


def _print_batch_summary(results: list[dict], elapsed: float, out_dir: Path):
    """Print the final batch summary table."""
    processed = [r for r in results if not r.get("skipped") and "error" not in r]
    skipped   = [r for r in results if r.get("skipped")]
    errors    = [r for r in results if "error" in r]

    total_pts    = sum(r["n_points"] for r in processed)
    total_in_mb  = sum(r["in_mb"]    for r in results)
    total_out_mb = sum(r["out_mb"]   for r in results)
    saving       = 100 * (1 - total_out_mb / total_in_mb) if total_in_mb > 0 else 0

    print(f"\n\n{SEP}")
    print("  BATCH SUMMARY")
    print(f"{SEP}")
    print(f"  {'Filename':<48}  {'Points':>10}  {'In MB':>7}  {'Out MB':>7}  {'s':>6}  Status")
    print(f"  {'─'*48}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*10}")

    for r in results:
        name   = r["csv_path"].name[:48]
        pts    = f"{r['n_points']:>10,}" if r["n_points"] else f"{'—':>10}"
        in_mb  = f"{r['in_mb']:>7.1f}"
        out_mb = f"{r['out_mb']:>7.1f}"
        secs   = f"{r['elapsed_sec']:>6.1f}"
        if r.get("skipped"):
            status = "SKIPPED"
        elif "error" in r:
            status = "ERROR"
        else:
            status = "OK"
        print(f"  {name:<48}  {pts}  {in_mb}  {out_mb}  {secs}  {status}")

    print(f"  {'─'*48}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*10}")
    print(f"  {'TOTAL':<48}  {total_pts:>10,}  {total_in_mb:>7.1f}  {total_out_mb:>7.1f}  {elapsed:>6.1f}")
    print(f"\n  Files processed : {len(processed)}")
    print(f"  Files skipped   : {len(skipped)}")
    print(f"  Files with error: {len(errors)}")
    print(f"  Total points    : {total_pts:,}")
    print(f"  Size reduction  : {saving:.0f}%")
    print(f"  Total time      : {elapsed:.1f} s")
    print(f"\n  Output folder   : {out_dir.resolve()}")
    print(f"\n  DuckDB usage:")
    print(f"    meta: read_parquet('{out_dir}/*_meta.parquet', union_by_name=True)")
    print(f"    ts  : read_parquet('{out_dir}/*_ts.parquet',   union_by_name=True)")
    print(f"\n  QGIS: Layer → Add Layer → Add Vector Layer → any *_meta.parquet")
    print(f"{SEP}\n")

    if errors:
        print("  ERRORS:")
        for r in errors:
            print(f"    {r['csv_path'].name}: {r['error']}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_pair(meta_path: Path, ts_path: Path):
    """Quick sanity check on one output pair."""
    print(f"\n  Verifying {meta_path.stem}...")
    gdf = gpd.read_parquet(meta_path)
    ts  = pd.read_parquet(ts_path, columns=["pid"])

    pid_match = set(gdf["pid"]) == set(ts["pid"])
    orbit_ok  = "orbit" in gdf.columns
    geom_ok   = gdf.crs is not None

    print(f"    Rows   : meta={len(gdf):,}  ts={len(ts):,}  "
          f"PID match={'✓' if pid_match else '✗'}")
    print(f"    CRS    : {gdf.crs}  {'✓' if geom_ok else '✗'}")
    print(f"    Orbit  : {gdf['orbit'].value_counts().to_dict() if orbit_ok else '✗ missing'}")
    print(f"    Vel    : mean={gdf['mean_velocity'].mean():.2f}  "
          f"min={gdf['mean_velocity'].min():.2f}  "
          f"max={gdf['mean_velocity'].max():.2f}  mm/yr")
    print(f"    Coher  : mean={gdf['temporal_coherence'].mean():.3f}")
    if "max_gap_days" in gdf.columns:
        print(f"    Gap    : max={gdf['max_gap_days'].iloc[0]} days  "
              f"n_acq={gdf['n_acquisitions'].iloc[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert EGMS CSV files to GeoParquet (one pair per file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
  Each CSV → <name>_meta.parquet  (GeoParquet 1.0, QGIS-ready)
           → <name>_ts.parquet    (float32 time series)
  All pairs queried together in DuckDB via wildcard + union_by_name=True.

Orbit detection:
  90 < (track_angle %% 360) < 270  →  Descending ('D')
  otherwise                         →  Ascending  ('A')

Examples:
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --out-dir C:/data/processed
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --min-coherence 0.7
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --compression zstd
  python egms_to_geoparquet.py --input-dir C:/data/EGMS_raw --force
        """,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing raw EGMS .csv / .txt files"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output folder for Parquet files (default: <input-dir>/processed_data/)"
    )
    parser.add_argument(
        "--chunksize", type=int, default=200_000,
        help="Rows per reading chunk, lower = less RAM (default: 200000)"
    )
    parser.add_argument(
        "--compression", choices=["snappy", "zstd", "gzip", "none"],
        default="snappy",
        help="Parquet compression codec (default: snappy)"
    )
    parser.add_argument(
        "--min-coherence", type=float, default=None,
        help="Remove points with temporal_coherence < value (e.g. 0.6)"
    )
    parser.add_argument(
        "--abs-velocity-limit", type=float, default=None,
        help="Remove points with |mean_velocity| > value mm/yr (e.g. 200)"
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=(
            "Spatial bounding box in WGS84 decimal degrees — only points inside\n"
            "are written to Parquet. Output goes to a named subfolder of\n"
            "out_dir so multiple extracts can coexist.\n"
            "Example: --bbox 10.5 45.0 12.5 46.5"
        )
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process files even if output already exists"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the per-file verification step"
    )

    args      = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"ERROR: --input-dir '{input_dir}' is not a valid directory.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "processed_data")

    bbox = tuple(args.bbox) if args.bbox is not None else None
    results = convert_folder(
        input_dir=input_dir,
        out_dir=out_dir,
        chunksize=args.chunksize,
        compression=args.compression,
        min_coherence=args.min_coherence,
        abs_velocity_limit=args.abs_velocity_limit,
        bbox=bbox,
        force=args.force,
    )

    if not args.no_verify:
        print(f"\n{SEP}")
        print("  VERIFICATION")
        print(f"{SEP}")
        for r in results:
            if not r.get("skipped") and "error" not in r:
                verify_pair(r["meta_out"], r["ts_out"])

    print("\n✓ All done.\n")


if __name__ == "__main__":
    main()
