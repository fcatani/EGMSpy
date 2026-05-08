#!/usr/bin/env python3
"""
EGMS Bridge  (v1)
==================
Flask + DuckDB server that connects the Parquet database to the HTML viewer.

How it works
------------
  1. At startup, user specifies (or confirms) the DATA_ROOT folder containing
     all *_meta.parquet and *_ts.parquet files produced by egms_to_geoparquet.py
  2. DuckDB registers two virtual views spanning ALL pairs in that folder:
       all_metadata  ←  *_meta.parquet  (union_by_name=True)
       all_timeseries ← *_ts.parquet   (union_by_name=True)
  3. Flask listens on localhost:5000 and answers two endpoints:
       GET /get_points      → spatial + filter query → JSON point list
       GET /get_timeseries  → single PID → JSON time series
       GET /get_info        → DB stats → JSON summary
  4. Browser is opened automatically on viewer.html after 2 s

Endpoints
---------
  /get_points
    Parameters:
      min_lat, max_lat, min_lon, max_lon  (float)  viewport bounds
      thresh      (float, default 2.5)    |mean_velocity| >= thresh  [mm/yr]
      min_coh     (float, default 0.0)    temporal_coherence >= min_coh
      asc         (bool,  default true)   include ascending  points
      desc        (bool,  default true)   include descending points
      zoom        (int,   default 15)     current zoom level (for decimation)
      max_pts     (int,   default 150000) hard cap on returned points
    Returns: JSON array of {pid, latitude, longitude, mean_velocity, orbit,
                             temporal_coherence, max_gap_days}

  /get_timeseries
    Parameters:
      pid  (str)  unique point identifier
    Returns: JSON object {pid, dates: [...], values: [...]}
             dates and values arrays are aligned; null values = data gaps

  /get_info
    Returns: JSON with DB statistics (point counts, orbit split, velocity
             range, coherence range, date ranges across all files)

Performance notes
-----------------
  - DuckDB reads only the row groups that overlap the bbox (covering bbox index
    written by the converter ensures this is fast even for 10M+ point datasets)
  - Zoom-level decimation: at low zoom, TABLESAMPLE reduces point density so
    Leaflet canvas never receives more than max_pts points regardless of area
  - All queries run in-memory on the DuckDB connection — no disk I/O per request
    beyond the initial Parquet row-group reads
  - union_by_name=True handles files with different date columns (different
    EGMS tracks/time windows) — missing dates returned as NULL

Usage
-----
  python bridge.py                          # prompts for DATA_ROOT interactively
  python bridge.py --data-dir C:/data/proc  # specify folder directly
  python bridge.py --data-dir C:/data/proc --port 5001 --no-browser

Requirements (conda env: egms)
-------------------------------
  conda install -c conda-forge flask flask-cors
  (duckdb, pandas, pathlib already in env)
"""

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path
from threading import Timer

import duckdb
import geopandas as gpd
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "═" * 65
SEP2 = "─" * 65

# Zoom-level decimation table:
# At low zoom levels the viewport covers a huge area — subsample to avoid
# sending 500k+ points to Leaflet canvas.
# Format: {zoom_level: keep_1_in_N}
# Class colour map — shared between bridge (for /get_info) and viewer
CLASS_COLORS = {
    "stable":        "#2ecc71",   # green
    "noisy":         "#95a5a6",   # grey
    "linear":        "#3498db",   # blue
    "accel":         "#e74c3c",   # red
    "decel":         "#f39c12",   # orange
    "variable":      "#9b59b6",   # purple
    "jump":          "#e67e22",   # dark orange
    "other":         "#1abc9c",   # teal
    "unclassified":  "#555555",   # dark grey
}


# ── Spatial grid thinning ──────────────────────────────────────────────────────
# Cell size in decimal degrees per zoom level.
# At zoom Z, one cell = GRID_DEG[Z] × GRID_DEG[Z] degrees.
# Only the highest-coherence point per cell is returned, giving uniform
# spatial coverage regardless of data density distribution.
# Cell size halves roughly every 2 zoom levels → 4× more points shown.
GRID_DEG = {
    0:  4.000,   # ~450 km — continents
    1:  2.000,
    2:  1.000,
    3:  0.500,
    4:  0.250,
    5:  0.120,   # ~13 km
    6:  0.060,
    7:  0.030,
    8:  0.015,   # ~1.5 km
    9:  0.008,
    10: 0.004,   # ~400 m
    11: 0.002,
    12: 0.001,   # ~100 m  (≈ EGMS PS spacing — all points visible)
    13: 0.0005,
    14: 0.0001,
    15: 0.00001, # show everything
}

# Hard cap on returned points — canvas renderer handles 200k+ comfortably
MAX_POINTS = 200_000

# HTML viewer filename (must be in same folder as bridge.py)
VIEWER_HTML = "viewer.html"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def find_parquet_files(data_root: Path) -> tuple[list[Path], list[Path]]:
    """Recursively find all _meta.parquet and _ts.parquet files under data_root."""
    meta_files = sorted(data_root.rglob("*_meta.parquet"))
    ts_files   = sorted(data_root.rglob("*_ts.parquet"))
    return meta_files, ts_files


def init_database(data_root: Path) -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB connection and register two views spanning
    all Parquet pairs found under data_root.
    """
    meta_files, ts_files = find_parquet_files(data_root)

    if not meta_files:
        raise FileNotFoundError(
            f"No *_meta.parquet files found under {data_root}\n"
            "Run egms_to_geoparquet.py first to produce the Parquet files."
        )

    print(f"\n{SEP}")
    print("  EGMS Bridge  —  DuckDB Initialisation")
    print(f"{SEP}")
    print(f"  Data root    : {data_root}")
    print(f"  Meta files   : {len(meta_files)}")
    print(f"  TS files     : {len(ts_files)}")
    print(f"{SEP2}")
    for mf in meta_files:
        tf = mf.parent / mf.name.replace("_meta.parquet", "_ts.parquet")
        tf_ok = "✓" if tf.exists() else "✗ (ts missing!)"
        sz_meta = mf.stat().st_size / 1_000_000
        sz_ts   = tf.stat().st_size / 1_000_000 if tf.exists() else 0
        print(f"  {mf.stem.replace('_meta',''):<50}  "
              f"meta={sz_meta:5.1f}MB  ts={sz_ts:6.1f}MB  {tf_ok}")

    # ── Connect and create views ──────────────────────────────────────────
    con = duckdb.connect(database=":memory:")

    # Use POSIX paths (forward slashes) — required on Windows too for DuckDB
    data_posix = Path(data_root).as_posix()

    print(f"\n  Creating DuckDB views (union_by_name=True)...")

    con.execute(f"""
        CREATE VIEW all_metadata AS
        SELECT * FROM read_parquet(
            '{data_posix}/**/*_meta.parquet',
            union_by_name = True,
            hive_partitioning = False
        )
    """)

    con.execute(f"""
        CREATE VIEW all_timeseries AS
        SELECT * FROM read_parquet(
            '{data_posix}/**/*_ts.parquet',
            union_by_name = True,
            hive_partitioning = False
        )
    """)

    # ── Quick DB diagnostics ──────────────────────────────────────────────
    print(f"  Running diagnostics...\n")

    total_pts = con.execute("SELECT COUNT(*) FROM all_metadata").fetchone()[0]
    print(f"  Total points   : {total_pts:>12,}")

    orbit_df = con.execute(
        "SELECT orbit, COUNT(*) as n FROM all_metadata GROUP BY orbit ORDER BY orbit"
    ).df()
    for _, row in orbit_df.iterrows():
        label = {"A": "Ascending", "D": "Descending", "U": "Unknown"}.get(
            str(row["orbit"]), str(row["orbit"])
        )
        print(f"  Orbit {label:<12}: {int(row['n']):>12,}")

    vel_stats = con.execute(
        "SELECT MIN(mean_velocity), MAX(mean_velocity), AVG(mean_velocity) "
        "FROM all_metadata"
    ).fetchone()
    print(f"  Velocity       : min={vel_stats[0]:.1f}  "
          f"max={vel_stats[1]:.1f}  mean={vel_stats[2]:.1f}  mm/yr")

    coh_stats = con.execute(
        "SELECT MIN(temporal_coherence), MAX(temporal_coherence) FROM all_metadata"
    ).fetchone()
    print(f"  Coherence      : min={coh_stats[0]:.3f}  max={coh_stats[1]:.3f}")

    # Detect date range from column names
    try:
        ts_cols_df = con.execute("DESCRIBE all_timeseries").df()
        import re as _re
        _DP = _re.compile(r'^D?(\d{8})$')
        def _bare(c):
            m = _DP.match(str(c).strip())
            return m.group(1) if m else None
        date_cols = sorted(
            [c for c in ts_cols_df["column_name"].tolist() if _bare(c)],
            key=lambda c: _bare(c)
        )
        if date_cols:
            print(f"  Date range     : {date_cols[0]} → {date_cols[-1]}"
                  f"  ({len(date_cols)} dates across all files)")
        else:
            print("  Date range     : ⚠ no date columns detected in TS files")
            date_cols = []
    except Exception as e:
        print(f"  Date range     : ⚠ could not read TS schema: {e}")
        date_cols = []

    print(f"\n{SEP}")
    print(f"  ✓ Database ready  —  {total_pts:,} points loaded")
    print(f"{SEP}\n")

    # Detect whether classification columns are present
    try:
        con.execute("SELECT class_1 FROM all_metadata LIMIT 0")
        has_cls = True
        print("  Classification columns detected in metadata ✓")
    except Exception:
        has_cls = False

    # Detect whether clustering has been run
    try:
        con.execute("SELECT cluster_id FROM all_metadata LIMIT 0")
        has_clust = True
        print("  Cluster columns detected in metadata ✓")
    except Exception:
        has_clust = False

    # Detect corrected time series files
    corr_files = sorted(Path(data_root).rglob("*_ts_corrected.parquet"))
    has_corr   = len(corr_files) > 0
    if has_corr:
        print(f"  Corrected TS files detected: {len(corr_files)} ✓")
    con_corr = None
    if has_corr:
        try:
            con_corr = duckdb.connect(":memory:")
            dp_corr  = Path(data_root).as_posix()
            con_corr.execute(f"""
                CREATE VIEW all_timeseries_corrected AS
                SELECT * FROM read_parquet(
                    '{dp_corr}/**/*_ts_corrected.parquet',
                    union_by_name = True,
                    hive_partitioning = False
                )
            """)
        except Exception as e:
            print(f"  ⚠  Could not open corrected TS view: {e}")
            con_corr  = None
            has_corr  = False

    return con, total_pts, date_cols if date_cols else [], has_cls, has_clust, has_corr, con_corr


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, expose_headers=["X-Total-Count", "X-Rendered-Count"])

# Module-level globals set during init (avoids passing con through Flask context)
_con          = None
_total_pts    = 0
_date_cols    = []
_data_root    = None
_has_classes  = False
_has_clusters = False
_has_subclusters = False            # True once egms_subcluster.py has been run
_cluster_poly_path: Path | None = None
_subcluster_poly_path: Path | None = None
_has_corrected = False
_con_corr      = None


# ── /get_points ───────────────────────────────────────────────────────────────

@app.route("/get_points", methods=["GET"])
def get_points():
    """
    Spatial + filter query with spatial grid thinning.

    Query params:
      min_lat, max_lat, min_lon, max_lon  — viewport bounds (required)
      thresh     — |mean_velocity| >= thresh  mm/yr  (default 2.5)
      min_coh    — temporal_coherence >= min_coh     (default 0.0)
      asc        — include ascending  ('true'/'false', default 'true')
      desc       — include descending ('true'/'false', default 'true')
      zoom       — current Leaflet zoom level (default 14)
      max_pts    — hard cap on returned points (default 200000)

    Thinning strategy (replaces old LIMIT-only decimation):
      Each viewport is divided into a grid of GRID_DEG[zoom] × GRID_DEG[zoom]
      degree cells.  Within each cell only the point with the highest
      temporal_coherence is returned.  This gives uniform spatial coverage at
      all zoom levels — no bias toward storage-order clusters — and always
      returns the most reliable representative per area.
      At zoom ≥ 12 the cell is smaller than typical EGMS PS spacing so all
      points are effectively returned.
    """
    t0 = time.perf_counter()
    try:
        # ── Parse parameters ─────────────────────────────────────────────
        min_lat  = float(request.args.get("min_lat",  -90))
        max_lat  = float(request.args.get("max_lat",   90))
        min_lon  = float(request.args.get("min_lon", -180))
        max_lon  = float(request.args.get("max_lon",  180))
        thresh   = float(request.args.get("thresh",   2.5))
        min_coh  = float(request.args.get("min_coh",  0.0))
        asc_on   = request.args.get("asc",  "true").lower() == "true"
        desc_on  = request.args.get("desc", "true").lower() == "true"
        zoom     = int(request.args.get("zoom",   14))
        max_pts  = int(request.args.get("max_pts", MAX_POINTS))

        # ── Orbit filter ──────────────────────────────────────────────────
        orbit_clauses = []
        if asc_on:  orbit_clauses.append("'A'")
        if desc_on: orbit_clauses.append("'D'")
        if not orbit_clauses:
            return jsonify([])
        orbit_in = ", ".join(orbit_clauses)

        # ── Cell size for this zoom level ─────────────────────────────────
        cell = GRID_DEG.get(min(zoom, 15), 0.00001)

        # ── Optional columns ──────────────────────────────────────────────
        try:
            _con.execute("SELECT max_gap_days FROM all_metadata LIMIT 0")
            gap_col = "COALESCE(CAST(max_gap_days AS INTEGER), 0)"
        except Exception:
            gap_col = "0"

        class_col = ""
        if _has_classes:
            try:
                _con.execute("SELECT class_1 FROM all_metadata LIMIT 0")
                class_col = ", COALESCE(CAST(class_1 AS VARCHAR), 'unclassified') AS class_1"
                # v3 descriptor flags — add silently if present, default false otherwise
                for flag_col in ("periodic", "jumpy", "variable_flag", "noisy_trend"):
                    try:
                        _con.execute(f"SELECT {flag_col} FROM all_metadata LIMIT 0")
                        class_col += f", COALESCE(CAST({flag_col} AS BOOLEAN), false) AS {flag_col}"
                    except Exception:
                        class_col += f", false AS {flag_col}"
            except Exception:
                class_col = ""

        # ── Single query: thinned points + total count via CTE ───────────────
        # Both the spatial-grid-thinned rows AND the raw count are computed in
        # one DuckDB execution — eliminating the second round-trip that caused
        # concurrent-connection conflicts when requests were aborted mid-flight.
        #
        # The CTE structure:
        #   filtered  — bbox + quality filter (shared base, scanned once)
        #   ranked    — ROW_NUMBER per grid cell (pick best-coherence point)
        #   total_cte — COUNT(*) over filtered (pre-thinning)
        #   thinned   — WHERE rn=1 LIMIT max_pts (the rendered subset)
        #
        # DuckDB executes this as a single query plan; the filtered CTE is
        # materialised once and reused by both ranked and total_cte branches.
        query = f"""
            WITH filtered AS (
                SELECT *
                FROM all_metadata
                WHERE latitude              BETWEEN {min_lat} AND {max_lat}
                  AND longitude             BETWEEN {min_lon} AND {max_lon}
                  AND ABS(mean_velocity)        >= {thresh}
                  AND temporal_coherence        >= {min_coh}
                  AND CAST(orbit AS VARCHAR) IN ({orbit_in})
            ),
            total_cte AS (
                SELECT COUNT(*) AS total_count FROM filtered
            ),
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            FLOOR(latitude  / {cell}),
                            FLOOR(longitude / {cell})
                        ORDER BY temporal_coherence DESC
                    ) AS _rn
                FROM filtered
            ),
            thinned AS (
                SELECT * FROM ranked WHERE _rn = 1 LIMIT {max_pts}
            )
            SELECT
                t.pid,
                t.latitude,
                t.longitude,
                ROUND(CAST(t.mean_velocity      AS DOUBLE), 2) AS mean_velocity,
                ROUND(CAST(t.temporal_coherence AS DOUBLE), 3) AS temporal_coherence,
                CAST(t.orbit AS VARCHAR)                        AS orbit,
                {gap_col}                                       AS max_gap_days
                {class_col},
                tc.total_count
            FROM thinned t
            CROSS JOIN total_cte tc
        """

        # Drop helper columns before serialising
        df           = _con.execute(query).df()
        total_in_view = int(df["total_count"].iloc[0]) if not df.empty else 0
        df           = df.drop(columns=["total_count", "_rn"], errors="ignore")
        elapsed      = (time.perf_counter() - t0) * 1000

        print(f"  /get_points  zoom={zoom:2d}  cell={cell}°  "
              f"bbox=[{min_lat:.3f},{min_lon:.3f}→{max_lat:.3f},{max_lon:.3f}]  "
              f"thresh=±{thresh}  coh≥{min_coh}  "
              f"→ {len(df):,} rendered / {total_in_view:,} total  ({elapsed:.0f} ms)")

        resp = Response(
            df.to_json(orient="records"),
            mimetype="application/json"
        )
        resp.headers["X-Total-Count"]    = str(total_in_view)
        resp.headers["X-Rendered-Count"] = str(len(df))
        return resp

    except Exception as e:
        import traceback
        print(f"  ERROR /get_points: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_timeseries ───────────────────────────────────────────────────────────

@app.route("/get_timeseries", methods=["GET"])
def get_timeseries():
    """
    Fetch time series for one PID.

    Returns JSON: { pid, dates: [...], values: [...], meta: {...} }
    dates and values are parallel arrays; null values = data gaps.
    """
    t0  = time.perf_counter()
    pid      = request.args.get("pid", "")
    use_corr = request.args.get("corrected", "false").lower() == "true"

    if not pid:
        return jsonify({"error": "pid parameter is required"}), 400

    try:
        # ── Fetch time series row ─────────────────────────────────────────
        # Sanitise PID to prevent injection (PIDs are alphanumeric in EGMS)
        pid_clean = "".join(c for c in pid if c.isalnum() or c in "-_")

        # Choose raw or corrected TS view
        if use_corr and _has_corrected and _con_corr is not None:
            ts_view, ts_con = "all_timeseries_corrected", _con_corr
        else:
            ts_view, ts_con = "all_timeseries", _con
            use_corr = False

        ts_df = ts_con.execute(
            f"SELECT * FROM {ts_view} WHERE pid = '{pid_clean}'"
        ).df()

        if ts_df.empty:
            return jsonify({"error": f"PID '{pid_clean}' not found"}), 404

        # ── Fetch metadata for this point ────────────────────────────────
        # Build column list defensively — optional columns may not exist
        # in files converted with older versions of the converter
        base_meta_cols = "pid, mean_velocity, temporal_coherence, CAST(orbit AS VARCHAR) as orbit, latitude, longitude, height, rmse, acceleration, seasonality"
        try:
            _con.execute("SELECT max_gap_days FROM all_metadata LIMIT 0")
            extra_cols = ", COALESCE(CAST(max_gap_days AS INTEGER),0) AS max_gap_days, COALESCE(CAST(n_acquisitions AS INTEGER),0) AS n_acquisitions"
        except Exception:
            extra_cols = ", 0 AS max_gap_days, 0 AS n_acquisitions"
        # Classification columns (optional — only present after egms_classify.py)
        try:
            _con.execute("SELECT class_1 FROM all_metadata LIMIT 0")
            extra_cols += (
                ", COALESCE(CAST(class_1 AS VARCHAR),'unclassified') AS class_1"
                ", COALESCE(CAST(class_2 AS VARCHAR),'unclassified') AS class_2"
                ", COALESCE(CAST(class_prob_1 AS DOUBLE), 0.0)       AS class_prob_1"
                ", COALESCE(CAST(periodic AS BOOLEAN), false)         AS periodic"
            )
            # v3 descriptor flags (optional — only present after egms_classify v3)
            for flag_col in ("jumpy", "variable_flag", "noisy_trend"):
                try:
                    _con.execute(f"SELECT {flag_col} FROM all_metadata LIMIT 0")
                    extra_cols += f", COALESCE(CAST({flag_col} AS BOOLEAN), false) AS {flag_col}"
                except Exception:
                    extra_cols += f", false AS {flag_col}"
        except Exception:
            extra_cols += ", 'unclassified' AS class_1, 'unclassified' AS class_2, 0.0 AS class_prob_1, false AS periodic, false AS jumpy, false AS variable_flag, false AS noisy_trend"

        meta_df = _con.execute(f"""
            SELECT {base_meta_cols}{extra_cols}
            FROM all_metadata
            WHERE pid = '{pid_clean}'
            LIMIT 1
        """).df()

        # ── Extract date columns and values ───────────────────────────────
        ts_row = ts_df.iloc[0]

        import re as _re2
        _DP2 = _re2.compile(r'^D?(\d{8})$')
        def _bare2(c):
            m = _DP2.match(str(c).strip())
            return m.group(1) if m else None

        all_ts_cols = sorted(
            [c for c in ts_df.columns if _bare2(c)],
            key=lambda c: _bare2(c)
        )

        # Diagnostic
        if all_ts_cols:
            sample_vals = []
            for c in all_ts_cols[:3]:
                try: sample_vals.append(round(float(ts_row[c]), 2))
                except: sample_vals.append(None)
            print(f"  TS cols found: {len(all_ts_cols)}  "
                  f"range: {_bare2(all_ts_cols[0])} → {_bare2(all_ts_cols[-1])}")
            print(f"  TS sample: {[_bare2(c) for c in all_ts_cols[:3]]} → {sample_vals}")
        else:
            print(f"  WARNING: no date cols found! "
                  f"Columns: {[str(c) for c in ts_df.columns[:10]]}")

        dates  = []
        values = []
        for col in all_ts_cols:
            bare = _bare2(col)
            dates.append(bare)          # always bare YYYYMMDD
            try:
                val  = ts_row[col]
                fval = float(val)
                values.append(None if pd.isna(fval) else round(fval, 2))
            except (TypeError, ValueError, KeyError):
                values.append(None)

        # ── Build metadata dict ───────────────────────────────────────────
        meta_dict = {}
        if not meta_df.empty:
            row = meta_df.iloc[0]
            meta_dict = {
                "mean_velocity":      round(float(row.get("mean_velocity",      0)), 2),
                "temporal_coherence": round(float(row.get("temporal_coherence", 0)), 3),
                "orbit":              str(row.get("orbit", "?")),
                "latitude":           round(float(row.get("latitude",           0)), 6),
                "longitude":          round(float(row.get("longitude",          0)), 6),
                "height":             round(float(row.get("height",             0)), 1),
                "rmse":               round(float(row.get("rmse",               0)), 2),
                "acceleration":       round(float(row.get("acceleration",       0)), 3),
                "seasonality":        round(float(row.get("seasonality",        0)), 3),
                "max_gap_days":       int(row.get("max_gap_days",   0) or 0),
                "n_acquisitions":     int(row.get("n_acquisitions", 0) or 0),
                "class_1":            str(row.get("class_1",    "unclassified")),
                "class_2":            str(row.get("class_2",    "unclassified")),
                "class_prob_1":       float(row.get("class_prob_1", 0.0) or 0.0),
                "periodic":           bool(row.get("periodic",      False)),
                "jumpy":              bool(row.get("jumpy",         False)),
                "variable_flag":      bool(row.get("variable_flag", False)),
                "noisy_trend":        bool(row.get("noisy_trend",   False)),
            }

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  /get_timeseries  pid={pid_clean}  "
              f"→ {len(dates)} dates  ({elapsed:.0f} ms)")

        # Fetch gap_info from metadata if available
        gap_info_list = []
        try:
            gi_row = _con.execute(
                f"SELECT gap_info FROM all_metadata WHERE pid = '{pid_clean}' LIMIT 1"
            ).fetchone()
            if gi_row and gi_row[0]:
                import json as _json
                gap_info_list = _json.loads(str(gi_row[0]))
        except Exception:
            pass

        return jsonify({
            "pid":       pid_clean,
            "dates":     dates,
            "values":    values,
            "meta":      meta_dict,
            "corrected": use_corr,
            "gap_info":  gap_info_list,
        })

    except Exception as e:
        import traceback
        print(f"  ERROR /get_timeseries: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_info ─────────────────────────────────────────────────────────────────

@app.route("/get_info", methods=["GET"])
def get_info():
    """
    Return database summary statistics.
    Called by viewer on startup to populate the info panel.
    """
    try:
        total = _con.execute("SELECT COUNT(*) FROM all_metadata").fetchone()[0]

        orbit_df = _con.execute(
            "SELECT orbit, COUNT(*) as n FROM all_metadata GROUP BY orbit"
        ).df()
        orbit_dict = dict(zip(orbit_df["orbit"].astype(str), orbit_df["n"].astype(int)))

        vel_row = _con.execute(
            "SELECT MIN(mean_velocity), MAX(mean_velocity), AVG(mean_velocity), "
            "STDDEV(mean_velocity) FROM all_metadata"
        ).fetchone()

        coh_row = _con.execute(
            "SELECT MIN(temporal_coherence), MAX(temporal_coherence) FROM all_metadata"
        ).fetchone()

        bbox_row = _con.execute(
            "SELECT MIN(longitude), MAX(longitude), MIN(latitude), MAX(latitude) "
            "FROM all_metadata"
        ).fetchone()

        return jsonify({
            "total_points":   int(total),
            "orbit_counts":   orbit_dict,
            "velocity": {
                "min":    round(float(vel_row[0]), 2),
                "max":    round(float(vel_row[1]), 2),
                "mean":   round(float(vel_row[2]), 2),
                "stddev": round(float(vel_row[3]), 2),
            },
            "coherence": {
                "min": round(float(coh_row[0]), 3),
                "max": round(float(coh_row[1]), 3),
            },
            "bbox": {
                "min_lon": round(float(bbox_row[0]), 4),
                "max_lon": round(float(bbox_row[1]), 4),
                "min_lat": round(float(bbox_row[2]), 4),
                "max_lat": round(float(bbox_row[3]), 4),
            },
            "date_range": {
                "start": _date_cols[0]  if _date_cols else None,
                "end":   _date_cols[-1] if _date_cols else None,
                "n_dates": len(_date_cols),
            },
            "data_root":       str(_data_root),
            "has_classes":     _has_classes,
            "class_colors":    CLASS_COLORS if _has_classes else {},
            "has_clusters":    _has_clusters,
            "has_subclusters": _has_subclusters,
            "has_corrected":   _has_corrected,
        })

    except Exception as e:
        print(f"  ERROR /get_info: {e}")
        return jsonify({"error": str(e)}), 500


# ── /get_clusters ────────────────────────────────────────────────────────────

@app.route("/get_clusters", methods=["GET"])
def get_clusters():
    """
    Return all cluster polygons as a GeoJSON FeatureCollection.
    Reads clusters_polygons.parquet (WGS84 GeoParquet written by egms_cluster.py).
    """
    t0 = time.perf_counter()
    try:
        if not _cluster_poly_path or not _cluster_poly_path.exists():
            return jsonify({"error": "No cluster polygons found. Run egms_cluster.py first."}), 404

        gdf = gpd.read_parquet(_cluster_poly_path)

        # Ensure WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Use geopandas .to_json() — correctly serialises numpy float32/int64/bool
        # __geo_interface__ returns raw numpy scalars which Flask jsonify cannot encode
        geojson_str = gdf.to_json(na="drop", show_bbox=True)

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  /get_clusters  → {len(gdf)} polygons  ({elapsed:.0f} ms)")
        # Return pre-serialised JSON string directly (avoids double-encoding)
        from flask import Response
        return Response(geojson_str, mimetype="application/json")

    except Exception as e:
        import traceback
        print(f"  ERROR /get_clusters: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_subclusters ──────────────────────────────────────────────────────────

@app.route("/get_subclusters", methods=["GET"])
def get_subclusters():
    """
    Return all sub-cluster polygons as a GeoJSON FeatureCollection.
    Reads subclusters_polygons.gpkg written by egms_subcluster.py.
    Falls back to GeoParquet if GPKG not available.
    """
    t0 = time.perf_counter()
    try:
        if not _subcluster_poly_path or not _subcluster_poly_path.exists():
            return jsonify({"error": "No sub-cluster polygons found. Run sub-clustering first."}), 404

        path = _subcluster_poly_path
        if path.suffix == ".gpkg":
            gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_parquet(path)

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geojson_str = gdf.to_json(na="drop", show_bbox=True)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  /get_subclusters → {len(gdf)} polygons  ({elapsed:.0f} ms)")
        return Response(geojson_str, mimetype="application/json")

    except Exception as e:
        import traceback
        print(f"  ERROR /get_subclusters: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /health ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint — viewer polls this on startup."""
    return jsonify({"status": "ok", "points": _total_pts})


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def prompt_data_root(default: Path | None = None) -> Path:
    """
    Interactive prompt for DATA_ROOT folder.
    Falls back to default if user just presses Enter.
    """
    print(f"\n{SEP}")
    print("  EGMS Bridge — Data Folder Setup")
    print(f"{SEP}")

    if default and default.is_dir():
        print(f"  Default path: {default}")
        user_input = input(
            f"  Press Enter to use default, or type a new path:\n  > "
        ).strip()
        return Path(user_input) if user_input else default
    else:
        while True:
            user_input = input(
                "  Enter the path to your processed_data folder\n"
                "  (containing *_meta.parquet files):\n  > "
            ).strip()
            p = Path(user_input)
            if p.is_dir():
                return p
            print(f"  ✗ Path not found: {p}  — please try again.")


def open_browser(port: int):
    """Open the viewer HTML in the default browser after a short delay."""
    html_path = Path(__file__).parent / VIEWER_HTML
    if html_path.exists():
        webbrowser.open(f"file://{html_path.resolve()}")
    else:
        print(f"\n  ⚠ viewer.html not found at {html_path}")
        print(f"    Place viewer.html in the same folder as bridge.py")
        print(f"    Bridge is still running — open viewer.html manually.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _con, _total_pts, _date_cols, _data_root, _has_classes, _has_clusters, _cluster_poly_path, _has_corrected, _con_corr

    parser = argparse.ArgumentParser(
        description="EGMS Bridge — Flask/DuckDB server for the HTML viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bridge.py                               # interactive folder prompt
  python bridge.py --data-dir C:/data/processed  # specify folder directly
  python bridge.py --data-dir C:/data/processed --port 5001
  python bridge.py --data-dir C:/data/processed --no-browser
        """,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Folder containing *_meta.parquet and *_ts.parquet files"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port for the Flask server (default: 5000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the browser automatically"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1 = localhost only)"
    )
    args = parser.parse_args()

    # ── Resolve data folder ───────────────────────────────────────────────
    if args.data_dir:
        data_root = Path(args.data_dir)
        if not data_root.is_dir():
            print(f"ERROR: --data-dir '{data_root}' is not a valid directory.")
            sys.exit(1)
    else:
        # Try to find a sensible default (processed_data sibling folder)
        default_guess = Path(__file__).parent / "processed_data"
        data_root = prompt_data_root(
            default=default_guess if default_guess.is_dir() else None
        )

    _data_root = data_root

    # ── Initialise DuckDB ─────────────────────────────────────────────────
    try:
        global _has_clusters, _cluster_poly_path, _has_corrected, _con_corr
        global _has_subclusters, _subcluster_poly_path
        _con, _total_pts, _date_cols, _has_classes, _has_clusters, _has_corrected, _con_corr = init_database(data_root)

        # Locate cluster polygon parquet if it exists
        poly_candidate = data_root / "clusters_polygons.parquet"
        if poly_candidate.exists():
            _cluster_poly_path = poly_candidate
            print(f"  Cluster polygons found: {poly_candidate.name} ✓")
        else:
            _cluster_poly_path = None

        # Locate sub-cluster polygon file (prefer GPKG, fall back to parquet)
        sub_gpkg = data_root / "subclusters_polygons.gpkg"
        sub_pq   = data_root / "subclusters_polygons.parquet"
        if sub_gpkg.exists():
            _subcluster_poly_path = sub_gpkg
            _has_subclusters = True
            print(f"  Sub-cluster polygons found: {sub_gpkg.name} ✓")
        elif sub_pq.exists():
            _subcluster_poly_path = sub_pq
            _has_subclusters = True
            print(f"  Sub-cluster polygons found: {sub_pq.name} ✓")
        else:
            _subcluster_poly_path = None
            _has_subclusters = False

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # ── Auto-open browser ─────────────────────────────────────────────────
    if not args.no_browser:
        Timer(2.0, open_browser, args=[args.port]).start()

    # ── Start Flask ───────────────────────────────────────────────────────
    print(f"\n  Starting Flask server on http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")
    print(f"{SEP}")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,   # must be False to avoid double DuckDB init
        threaded=True,        # handle concurrent requests (pan + TS click)
    )


if __name__ == "__main__":
    main()
