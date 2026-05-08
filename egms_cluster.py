#!/usr/bin/env python3
"""
EGMS Spatial Cluster Detector  (v1)
=====================================
Finds spatially coherent clusters of deforming InSAR points using a
velocity-weighted 3D DBSCAN algorithm, then:
  - Writes cluster_id to every point in the *_meta.parquet files
  - Generates a cluster polygon GeoParquet (convex hull + buffer)
  - Generates a cluster summary CSV

Algorithm
---------
  Coordinates fed to DBSCAN:
    X  = easting  (metres, UTM)
    Y  = northing (metres, UTM)
    Vw = mean_velocity × (eps_m / eps_vel)

  The velocity axis is rescaled so that a velocity difference of eps_vel mm/yr
  is "equivalent" to a spatial separation of eps_m metres inside the
  epsilon-ball. This means two points are neighbours if they are BOTH
  within eps_m metres spatially AND within eps_vel mm/yr in velocity.

  After DBSCAN, each cluster is characterised by its dominant TS class
  (majority vote among class_1 labels). Clusters below min_class_purity
  are flagged as "mixed" but kept.

Output files (written to data_dir)
-----------------------------------
  clusters_points.parquet   — metadata parquet updated with cluster_id column
                              (one row per point; unclassified → "NOISE")
  clusters_polygons.parquet — GeoParquet with one convex-hull+buffer polygon
                              per cluster, with full attribute table
  clusters_summary.csv      — human-readable cluster table

Usage
-----
  python egms_cluster.py --data-dir ./processed_data
  python egms_cluster.py --data-dir ./processed_data \\
      --min-vel 10 --min-coh 0.3 --eps-m 100 --eps-vel 5 \\
      --min-samples 5 --buffer-m 50

Requirements: pandas, numpy, geopandas, pyarrow, duckdb, scikit-learn, scipy, shapely, tqdm
"""

import argparse
import os
import json
import re
import sys
import time
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEP  = "═" * 68
SEP2 = "─" * 68

# Class colour map (mirrors bridge.py / viewer.html)
CLASS_COLORS = {
    "stable":        "#2ecc71",
    "noisy":         "#95a5a6",
    "linear":        "#3498db",
    "accel":         "#e74c3c",
    "decel":         "#f39c12",
    "variable":      "#9b59b6",
    "jump":          "#e67e22",
    "other":         "#1abc9c",
    "unclassified":  "#555555",
    "mixed":         "#ffffff",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_cluster_id(n: int) -> str:
    """Return a zero-padded cluster ID string, e.g. CL_0042."""
    return f"CL_{n:04d}"


def _convex_hull_polygon(east: np.ndarray, north: np.ndarray) -> Polygon | None:
    """
    Build Shapely Polygon from easting/northing arrays.
    Falls back to Point/LineString buffer for very small clusters.
    """
    pts = np.column_stack([east, north])
    n   = len(pts)
    if n == 0:
        return None
    if n == 1:
        return Point(pts[0]).buffer(1)
    if n == 2:
        from shapely.geometry import LineString
        return LineString(pts).buffer(1)
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        return Polygon(verts)
    except Exception:
        return MultiPoint(pts).convex_hull


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ELIGIBLE POINTS FROM ALL PARQUET FILES
# ══════════════════════════════════════════════════════════════════════════════

def load_eligible_points(
    data_dir:  Path,
    min_vel:   float,
    min_coh:   float,
    exclude_classes: list[str],
) -> pd.DataFrame:
    """
    Load pid, easting, northing, latitude, longitude, mean_velocity,
    temporal_coherence, class_1, class_2 from ALL *_meta.parquet files
    using DuckDB union_by_name=True.

    Filters applied:
      - |mean_velocity| >= min_vel
      - temporal_coherence >= min_coh
      - class_1 NOT IN exclude_classes  (skips 'unclassified' by default)
    """
    meta_files = sorted(data_dir.rglob("*_meta.parquet"))
    # Exclude the cluster polygon output file if it exists
    meta_files = [f for f in meta_files if "clusters_" not in f.name]
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.parquet files found under {data_dir}")

    con = duckdb.connect(":memory:")
    dp  = data_dir.as_posix()
    con.execute(f"""
        CREATE VIEW meta AS
        SELECT * FROM read_parquet('{dp}/**/*_meta.parquet', union_by_name=True)
    """)

    # Coherence / velocity distribution info
    stats = con.execute("""
        SELECT COUNT(*),
               MIN(temporal_coherence), MAX(temporal_coherence),
               MIN(ABS(mean_velocity)), MAX(ABS(mean_velocity))
        FROM meta
    """).fetchone()
    print(f"  Total points   : {stats[0]:,}")
    print(f"  Coherence      : {stats[1]:.3f} → {stats[2]:.3f}")
    print(f"  |Velocity|     : {stats[3]:.1f} → {stats[4]:.1f} mm/yr")

    # Build class filter
    excl_sql = "', '".join(exclude_classes)
    has_class_col = True
    try:
        con.execute("SELECT class_1 FROM meta LIMIT 0")
    except Exception:
        has_class_col = False
        print("  ⚠  No class_1 column — run egms_classify.py first")
        print("     Clustering will ignore class filter")

    class_filter = ""
    if has_class_col and exclude_classes:
        class_filter = f"AND COALESCE(CAST(class_1 AS VARCHAR),'unclassified') NOT IN ('{excl_sql}')"

    # EGMS easting/northing are always EPSG:3035 (ETRS89-LAEA Europe)
    # Verify the columns exist; fall back to lat/lon if missing
    has_laea = True
    try:
        con.execute("SELECT easting, northing FROM meta LIMIT 0")
    except Exception:
        has_laea = False
        print("  ⚠  No easting/northing columns — will use lat/lon directly")

    if has_laea:
        coord_cols = "easting, northing"
        print("  Coordinate CRS : EPSG:3035 (ETRS89-LAEA, native EGMS)")
    else:
        coord_cols = "longitude AS easting, latitude AS northing"
        print("  Coordinate CRS : EPSG:4326 fallback (no easting/northing col)")

    class_sel = ""
    class_sel2 = ""
    if has_class_col:
        class_sel  = ", COALESCE(CAST(class_1 AS VARCHAR),'unclassified') AS class_1"
        class_sel2 = ", COALESCE(CAST(class_2 AS VARCHAR),'unclassified') AS class_2"
        class_sel3 = ", COALESCE(CAST(class_prob_1 AS DOUBLE), 0.0) AS class_prob_1"
    else:
        class_sel  = ", 'unclassified' AS class_1"
        class_sel2 = ", 'unclassified' AS class_2"
        class_sel3 = ", 0.0 AS class_prob_1"

    query = f"""
        SELECT
            CAST(pid AS VARCHAR)                         AS pid,
            CAST({coord_cols.split(',')[0]} AS DOUBLE)   AS easting,
            CAST({coord_cols.split(',')[1].strip()} AS DOUBLE) AS northing,
            CAST(latitude  AS DOUBLE)                    AS latitude,
            CAST(longitude AS DOUBLE)                    AS longitude,
            ROUND(CAST(mean_velocity      AS DOUBLE), 2) AS mean_velocity,
            ROUND(CAST(temporal_coherence AS DOUBLE), 3) AS temporal_coherence
            {class_sel}{class_sel2}{class_sel3}
        FROM meta
        WHERE ABS(mean_velocity)  >= {min_vel}
          AND temporal_coherence  >= {min_coh}
          {class_filter}
    """
    df = con.execute(query).df()
    con.close()

    if not has_laea:
        # No easting/northing: approximate LAEA from lat/lon using geopandas
        print("  Projecting lat/lon → EPSG:3035 (LAEA)...")
        gdf_ll = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        gdf_laea = gdf_ll.to_crs(epsg=3035)
        df["easting"]  = gdf_laea.geometry.x
        df["northing"] = gdf_laea.geometry.y

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VELOCITY-WEIGHTED 3D DBSCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_dbscan(
    df:          pd.DataFrame,
    eps_m:       float,
    eps_vel:     float,
    min_samples: int,
) -> np.ndarray:
    """
    Run DBSCAN on (easting, northing, velocity_scaled) coordinates.

    The velocity axis is rescaled so that eps_vel mm/yr of velocity difference
    equals eps_m metres of spatial distance inside the epsilon ball:
        vel_weight = eps_m / eps_vel
        vel_coord  = mean_velocity × vel_weight

    The single DBSCAN epsilon is eps_m (in metres / equivalent units).
    This means a pair of points is a "neighbour" iff:
        sqrt( Δeast² + Δnorth² + (Δvel × w)² ) ≤ eps_m
    which is satisfied only when BOTH spatial AND velocity proximity hold.

    Returns: labels array (n_pts,), -1 = noise.
    """
    vel_weight = eps_m / max(eps_vel, 1e-9)

    X = np.column_stack([
        df["easting"].values,
        df["northing"].values,
        df["mean_velocity"].values * vel_weight,
    ])

    print(f"  Running DBSCAN on {len(X):,} points …")
    print(f"    eps_m={eps_m} m, eps_vel={eps_vel} mm/yr → vel_weight={vel_weight:.2f}")
    print(f"    Effective: neighbours within {eps_m} m AND {eps_vel} mm/yr")

    db = DBSCAN(
        eps=eps_m,
        min_samples=min_samples,
        algorithm='ball_tree',   # most efficient for 3D euclidean
        metric='euclidean',
        n_jobs=-1,
    ).fit(X)

    return db.labels_


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER CHARACTERISATION
# ══════════════════════════════════════════════════════════════════════════════

def characterise_clusters(
    df:             pd.DataFrame,
    raw_labels:     np.ndarray,
    min_purity:     float,
    buffer_m:       float,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    For each DBSCAN cluster (label >= 0):
      - Determine dominant class (majority vote on class_1)
      - Compute class purity = fraction of points with dominant class
      - Build convex hull polygon in UTM, then buffer by buffer_m
      - Compute area (m²), centroid
      - Assign cluster_id string (e.g. CL_0001)

    Returns:
      pts_df  — original df with cluster_id column added
      poly_gdf — GeoDataFrame of cluster polygons (UTM CRS)

    Performance note:
      cluster_id assignment is fully vectorised — a label→cid map is built
      once and applied in a single numpy operation, avoiding the O(N×C)
      cost of df.loc[mask, col] = val called C times over N rows.
    """
    df = df.copy()

    cluster_ids_raw = sorted(set(raw_labels[raw_labels >= 0]))
    n_clusters = len(cluster_ids_raw)
    print(f"  DBSCAN found {n_clusters} raw clusters  "
          f"({(raw_labels == -1).sum():,} noise points)")

    if n_clusters == 0:
        print("  ⚠  No clusters found. Try: lower --min-vel, increase --eps-m,"
              " decrease --min-samples")
        df["cluster_id"] = "NOISE"
        return df, gpd.GeoDataFrame()

    # ── Pre-extract numpy arrays for fast indexed access ──────────────────
    east_arr  = df["easting"].values
    north_arr = df["northing"].values
    vel_arr   = df["mean_velocity"].values
    coh_arr   = df["temporal_coherence"].values
    cls_arr   = df["class_1"].values

    # ── label → CL_XXXX mapping (built incrementally, applied once) ───────
    # raw_label (int) → cid string; -1 → "NOISE"
    label_to_cid: dict[int, str] = {-1: "NOISE"}

    polygon_rows = []
    cluster_num  = 1

    for raw_label in tqdm(cluster_ids_raw, desc="  Building polygons", leave=False):
        mask = raw_labels == raw_label   # boolean array, used only for indexing
        idx  = np.where(mask)[0]         # integer indices — much faster for numpy ops
        n    = len(idx)

        # ── Dominant class ────────────────────────────────────────────────
        cls_vals     = cls_arr[idx]
        unique, counts = np.unique(cls_vals, return_counts=True)
        top_i        = counts.argmax()
        dominant_cls = unique[top_i]
        purity       = counts[top_i] / n
        is_mixed     = purity < min_purity

        # ── Velocity stats ────────────────────────────────────────────────
        v      = vel_arr[idx]
        v_mean = float(v.mean())
        v_std  = float(v.std())
        v_min  = float(v.min())
        v_max  = float(v.max())

        # ── Coherence stats ───────────────────────────────────────────────
        coh_mean = float(coh_arr[idx].mean())

        # ── Convex hull + buffer ──────────────────────────────────────────
        east  = east_arr[idx]
        north = north_arr[idx]
        hull_poly = _convex_hull_polygon(east, north)
        if hull_poly is None:
            # Skip cluster but still map its label to NOISE
            label_to_cid[raw_label] = "NOISE"
            continue
        buffered = hull_poly.buffer(buffer_m, cap_style=1, join_style=1, resolution=32)

        # ── Centroid (UTM) ────────────────────────────────────────────────
        cx, cy  = buffered.centroid.x, buffered.centroid.y
        area_m2 = buffered.area

        # ── Register label→cid (actual assignment happens after loop) ─────
        cid = _make_cluster_id(cluster_num)
        label_to_cid[raw_label] = cid

        polygon_rows.append({
            "cluster_id":    cid,
            "n_points":      n,
            "dom_class":     dominant_cls if not is_mixed else "mixed",
            "class_purity":  round(purity, 3),
            "is_mixed":      is_mixed,
            "vel_mean":      round(v_mean, 2),
            "vel_std":       round(v_std,  2),
            "vel_min":       round(v_min,  2),
            "vel_max":       round(v_max,  2),
            "coh_mean":      round(coh_mean, 3),
            "area_m2":       round(area_m2, 1),
            "area_ha":       round(area_m2 / 10_000, 4),
            "centroid_e":    round(cx, 1),
            "centroid_n":    round(cy, 1),
            "color":         CLASS_COLORS.get(
                                dominant_cls if not is_mixed else "mixed",
                                "#cccccc"),
            "geometry":      buffered,
        })
        cluster_num += 1

    print(f"  Retained {len(polygon_rows)} clusters")

    # ── Vectorised cluster_id assignment — single pass over all points ────
    # Map raw integer labels → cid strings using a numpy vectorised lookup.
    # This replaces 28k+ individual df.loc[mask, col] = val calls which
    # trigger pandas CoW checks on every iteration (catastrophic at scale).
    cid_values = np.array(
        [label_to_cid.get(lbl, "NOISE") for lbl in raw_labels],
        dtype=object
    )
    df["cluster_id"] = cid_values

    # ── Build polygon GeoDataFrame ────────────────────────────────────────
    poly_gdf = gpd.GeoDataFrame(polygon_rows, crs=None)   # CRS set later
    return df, poly_gdf


# ══════════════════════════════════════════════════════════════════════════════
# WRITE CLUSTER_ID BACK TO META PARQUETS
# ══════════════════════════════════════════════════════════════════════════════

def _update_one_parquet(mf: Path, id_map: pd.Series) -> str:
    """Update a single meta parquet with cluster_id. Returns filename on success."""
    try:
        table = pq.read_table(mf)
        df    = table.to_pandas()
        if "cluster_id" in df.columns:
            df = df.drop(columns=["cluster_id"])
        df["cluster_id"] = df["pid"].astype(str).map(id_map).fillna("NOISE")
        tmp       = mf.with_suffix(".tmp.parquet")
        orig_meta = table.schema.metadata or {}
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        new_table = new_table.replace_schema_metadata(orig_meta)
        pq.write_table(new_table, tmp, compression="snappy")
        tmp.replace(mf)
        return f"  ✓ {mf.name}"
    except Exception as ex:
        return f"  ✗ {mf.name}: {ex}"


def update_meta_parquets(
    data_dir:   Path,
    pts_df:     pd.DataFrame,   # must contain pid, cluster_id
    n_jobs:     int = -1,
):
    """
    Update every *_meta.parquet file with the cluster_id column.
    Uses atomic temp-file rename + parallel writes for speed.
    At 300M points / 500+ files the write-back is I/O bound —
    parallelising across files gives near-linear speedup up to
    the number of physical disks (SSD: saturates at ~8-16 workers).
    """
    from joblib import Parallel, delayed as _delayed

    meta_files = sorted(data_dir.rglob("*_meta.parquet"))
    meta_files = [f for f in meta_files if "clusters_" not in f.name]

    id_map = pts_df.set_index("pid")["cluster_id"]

    n_cores  = os.cpu_count() or 1
    eff_jobs = n_cores if n_jobs == -1 else min(abs(n_jobs), n_cores)
    # Cap at 16 for write-back — SSD I/O saturates before CPU does
    eff_jobs = min(eff_jobs, 16)
    print(f"  Writing cluster_id to {len(meta_files)} parquet files "
          f"({eff_jobs} parallel writers) ...", flush=True)

    results = Parallel(n_jobs=eff_jobs, backend="threading")(
        _delayed(_update_one_parquet)(mf, id_map) for mf in meta_files
    )
    for msg in results:
        print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# INFER UTM EPSG FROM CENTROID
# ══════════════════════════════════════════════════════════════════════════════

# EGMS easting/northing coordinates are ALWAYS in EPSG:3035
# (ETRS89 / LAEA Europe) regardless of the geographic area of the tile.
# This is mandated by the EGMS product specification.
# Values are in metres with false origin at lon=10°E, lat=52°N.
EGMS_NATIVE_EPSG = 3035


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def _parquet_write_safe(gdf: gpd.GeoDataFrame, path: Path):
    """
    Write GeoDataFrame to GeoParquet with graceful version fallback.
    Tries progressively simpler calls until one succeeds.
    """
    import geopandas as _gpd
    gv = tuple(int(x) for x in _gpd.__version__.split(".")[:2])

    attempts = []
    if gv >= (0, 14):
        attempts.append(dict(compression="snappy", index=False,
                             write_covering_bbox=True, schema_version="1.0.0"))
    attempts.append(dict(compression="snappy", index=False,
                         schema_version="1.0.0"))
    attempts.append(dict(compression="snappy", index=False))
    attempts.append(dict(index=False))

    last_err = None
    for kwargs in attempts:
        try:
            gdf.to_parquet(path, **kwargs)
            # Verify: re-read and check row count matches
            import pyarrow.parquet as _pq
            written = _pq.read_metadata(str(path)).num_rows
            if written != len(gdf):
                raise RuntimeError(
                    f"Row count mismatch after write: wrote {written}, expected {len(gdf)}"
                )
            return   # success
        except Exception as e:
            last_err = e
            if path.exists():
                path.unlink()   # remove corrupt partial file
            continue

    raise RuntimeError(
        f"All GeoParquet write attempts failed. Last error: {last_err}\n"
        f"Try: conda install -c conda-forge geopandas pyarrow"
    )


def save_outputs(
    poly_gdf: gpd.GeoDataFrame,
    pts_df:   pd.DataFrame,
    data_dir: Path,
    src_epsg: int,
    export_shapefile: bool = False,
):
    """
    Save:
      1. clusters_polygons.parquet  — GeoParquet WGS84 (for bridge + QGIS)
      2. clusters_polygons.gpkg     — GeoPackage LAEA (for QGIS / ArcGIS)
      3. clusters_polygons.shp      — Shapefile WGS84 (optional, --shapefile)
      4. clusters_summary.csv       — human-readable table
    src_epsg: source CRS of polygon coordinates (EGMS = EPSG:3035 LAEA)
    """
    if poly_gdf.empty:
        print("  No clusters to save.")
        return

    # ── Assign native EGMS CRS (EPSG:3035 LAEA) then reproject to WGS84 ───
    poly_gdf = poly_gdf.set_crs(epsg=src_epsg)
    poly_wgs  = poly_gdf.to_crs(epsg=4326)

    # ── GeoParquet (WGS84) ────────────────────────────────────────────────
    poly_out = data_dir / "clusters_polygons.parquet"
    _parquet_write_safe(poly_wgs, poly_out)
    print(f"  ✓ GeoParquet    : {poly_out}")

    # ── GeoPackage (QGIS/ArcGIS) — LAEA for accurate area ────────────────
    gpkg_out = data_dir / "clusters_polygons.gpkg"
    poly_gdf.to_file(gpkg_out, driver="GPKG", layer="clusters")
    print(f"  ✓ GeoPackage    : {gpkg_out}")

    # ── Shapefile (WGS84, optional) ───────────────────────────────────────
    if export_shapefile:
        shp_out = data_dir / "clusters_polygons.shp"
        # Shapefile column names limited to 10 chars — truncate with dedup
        shp_gdf = poly_wgs.copy()
        col_map = {}
        seen    = set()
        for col in shp_gdf.columns:
            if col == "geometry":
                continue
            short = col[:10]
            if short in seen:
                short = col[:8] + f"{sum(1 for s in seen if s.startswith(col[:8])):02d}"
            seen.add(short)
            col_map[col] = short
        shp_gdf = shp_gdf.rename(columns=col_map)
        shp_gdf.to_file(shp_out, driver="ESRI Shapefile")
        print(f"  ✓ Shapefile     : {shp_out}  (WGS84, col names truncated to 10 chars)")

    # ── CSV summary ───────────────────────────────────────────────────────
    csv_cols = [c for c in poly_gdf.columns if c != "geometry"]
    csv_out  = data_dir / "clusters_summary.csv"
    poly_gdf[csv_cols].to_csv(csv_out, index=False)
    print(f"  ✓ CSV summary   : {csv_out}")

    # ── Stats printout ────────────────────────────────────────────────────
    n_cls = len(poly_gdf)
    n_pts_in_clusters = (pts_df["cluster_id"] != "NOISE").sum()
    n_noise           = (pts_df["cluster_id"] == "NOISE").sum()

    print(f"\n  Cluster statistics:")
    print(f"    Total clusters    : {n_cls}")
    print(f"    Points in clusters: {n_pts_in_clusters:,}")
    print(f"    Noise points      : {n_noise:,}")
    print(f"    Avg cluster size  : {n_pts_in_clusters/max(n_cls,1):.0f} pts")
    print(f"    Avg area          : {poly_gdf['area_m2'].mean():.0f} m²")

    print(f"\n  By dominant class:")
    cls_counts = poly_gdf.groupby("dom_class").agg(
        n_clusters=("cluster_id", "count"),
        total_pts =("n_points",   "sum"),
        mean_area =("area_m2",    "mean"),
    )
    for cls, row in cls_counts.iterrows():
        print(f"    {cls:<14}: {int(row.n_clusters):>4} clusters  "
              f"{int(row.total_pts):>7,} pts  "
              f"avg area {row.mean_area:.0f} m²")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EGMS velocity-weighted spatial clustering (3D DBSCAN).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data-dir",    required=True,
        help="Folder with *_meta.parquet files (output of egms_to_geoparquet.py)")

    # Point eligibility filters
    parser.add_argument("--min-vel",     type=float, default=2.5,
        help="Minimum |mean_velocity| to include in clustering (mm/yr, default 2.5 — "
             "tuned for high-coherence EGMS L3 data in Italy where meaningful deformation "
             "starts at ~2-3 mm/yr; use 5-10 for noisier L2b data)")
    parser.add_argument("--min-coh",     type=float, default=0.5,
        help="Minimum temporal_coherence to include (default 0.5 — appropriate for "
             "Italian EGMS L3 which is pre-filtered for high coherence; "
             "lower to 0.3 for L2b or noisier datasets)")
    parser.add_argument("--exclude-classes", nargs="*",
        default=["unclassified", "stable", "noisy"],
        help="Class names to exclude (default: unclassified stable noisy)")

    # DBSCAN parameters
    parser.add_argument("--eps-m",       type=float, default=100.0,
        help="Spatial epsilon in metres (default 100)")
    parser.add_argument("--eps-vel",     type=float, default=3.0,
        help="Velocity epsilon in mm/yr equivalent to eps-m (default 3.0 — "
             "tighter than the old 5.0 default to avoid merging adjacent slow-moving "
             "slopes with different rates; increase to 8-10 for jump-class dominated data)")
    parser.add_argument("--min-samples", type=int,   default=5,
        help="DBSCAN min_samples: min points to form a cluster (default 5)")

    # Polygon parameters
    parser.add_argument("--buffer-m",    type=float, default=20.0,
        help="Buffer distance around convex hull in metres (default 20)")
    parser.add_argument("--min-purity",  type=float, default=0.5,
        help="Min fraction of dominant class to label cluster as pure (default 0.5)")
    parser.add_argument("--shapefile",   action="store_true",
        help="Also export cluster polygons as ESRI Shapefile (.shp) in WGS84")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        print(f"ERROR: '{data_dir}' is not a valid directory")
        sys.exit(1)

    print(f"\n{SEP}")
    print("  EGMS SPATIAL CLUSTER DETECTOR  v1")
    print(f"{SEP}")
    print(f"  Data dir       : {data_dir}")
    print(f"\n  Point filters:")
    print(f"    min |velocity|  : {args.min_vel} mm/yr")
    print(f"    min coherence   : {args.min_coh}")
    print(f"    exclude classes : {args.exclude_classes}")
    print(f"\n  DBSCAN parameters:")
    print(f"    eps_m           : {args.eps_m} m")
    print(f"    eps_vel         : {args.eps_vel} mm/yr")
    print(f"    vel_weight      : {args.eps_m/args.eps_vel:.2f}  "
          f"(1 mm/yr ≡ {args.eps_m/args.eps_vel:.1f} m in the DBSCAN distance)")
    print(f"    min_samples     : {args.min_samples}")
    print(f"\n  Polygon parameters:")
    print(f"    buffer          : {args.buffer_m} m")
    print(f"    min_purity      : {args.min_purity:.0%}")
    print(f"    shapefile       : {'YES' if args.shapefile else 'NO'}")

    t0 = time.time()

    # ── Load eligible points ───────────────────────────────────────────────
    print(f"\n{SEP2}\n  Loading eligible points …\n{SEP2}")
    df = load_eligible_points(
        data_dir        = data_dir,
        min_vel         = args.min_vel,
        min_coh         = args.min_coh,
        exclude_classes = args.exclude_classes,
    )
    n_elig = len(df)
    print(f"  Eligible points: {n_elig:,}")

    if n_elig == 0:
        print("  ✗ No eligible points — adjust --min-vel, --min-coh, or --exclude-classes")
        sys.exit(1)

    if n_elig < args.min_samples:
        print(f"  ✗ Fewer eligible points ({n_elig}) than min_samples ({args.min_samples})")
        sys.exit(1)

    # EGMS easting/northing are always EPSG:3035 (ETRS89-LAEA Europe)
    src_epsg = EGMS_NATIVE_EPSG
    print(f"  Source CRS     : EPSG:{src_epsg} (ETRS89-LAEA — all EGMS products)")

    # ── DBSCAN ────────────────────────────────────────────────────────────
    print(f"\n{SEP2}\n  Running 3D DBSCAN …\n{SEP2}")
    raw_labels = run_dbscan(
        df          = df,
        eps_m       = args.eps_m,
        eps_vel     = args.eps_vel,
        min_samples = args.min_samples,
    )
    n_found = len(set(raw_labels[raw_labels >= 0]))
    n_noise = (raw_labels == -1).sum()
    print(f"  Raw clusters   : {n_found}")
    print(f"  Noise points   : {n_noise:,}  ({100*n_noise/n_elig:.1f}%)")

    # ── Characterise clusters and build polygons ───────────────────────────
    print(f"\n{SEP2}\n  Building cluster polygons …\n{SEP2}")
    pts_df, poly_gdf = characterise_clusters(
        df          = df,
        raw_labels  = raw_labels,
        min_purity  = args.min_purity,
        buffer_m    = args.buffer_m,
    )

    # ── Update metadata parquets with cluster_id ───────────────────────────
    print(f"\n{SEP2}\n  Writing cluster_id to metadata parquets …\n{SEP2}")
    update_meta_parquets(data_dir, pts_df)

    # ── Save polygon outputs ───────────────────────────────────────────────
    print(f"\n{SEP2}\n  Saving outputs …\n{SEP2}")
    save_outputs(poly_gdf, pts_df, data_dir, src_epsg,
                 export_shapefile=args.shapefile)

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  ✓ Clustering complete in {elapsed:.1f} s")
    print(f"  cluster_id written to *_meta.parquet files")
    print(f"  Polygon files in: {data_dir}")
    print(f"  Restart bridge.py to serve cluster data to the viewer")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
