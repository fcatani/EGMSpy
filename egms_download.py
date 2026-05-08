#!/usr/bin/env python3
"""
EGMS Data Downloader
====================
Downloads EGMS L2a/L2b CSV tiles from the Copernicus Land Monitoring Service
for a user-defined bounding box in WGS84 (lat/lon).

Requires EGMStoolkit installed in the same conda environment:
    git clone https://github.com/alexisInSAR/EGMStoolkit.git
    Add to site-packages path (see installation notes)

HOW TO GET YOUR TOKEN
---------------------
1. Go to https://egms.land.copernicus.eu/
2. Log in (free EU Copernicus account required)
3. Click on any tile download button
4. Right-click the download button → "Copy link address"
   OR use browser developer tools (F12 → Network tab → click download)
5. The token is the string after ?id= in the URL:
   https://egms.land.copernicus.eu/.../file.zip?id=a212e123be834582...
   Your token: a212e123be834582...
   Tokens expire after ~24 hours.

USAGE
-----
  # Interactive (prompts for all parameters):
  python egms_download.py

  # Command-line (fully automated):
  python egms_download.py \\
      --token YOUR_TOKEN \\
      --bbox 6.55 43.63 14.02 47.18 \\
      --product L2b \\
      --period 2019_2023 \\
      --out-dir ./po_valley_raw

  # Dry run (list tiles without downloading):
  python egms_download.py --token YOUR_TOKEN --bbox 6.55 43.63 14.02 47.18 --dry-run

BOUNDING BOX FORMAT
-------------------
  --bbox lon_min lat_min lon_max lat_max   (WGS84 decimal degrees)
  Example for Po Valley:  --bbox 6.55 43.63 14.02 47.18
  Example for NE Italy:   --bbox 11.0 45.5  13.8  46.8

PRODUCT LEVELS
--------------
  L2a  — Basic (Line-of-Sight, local reference)
  L2b  — GNSS-Calibrated (recommended)

PERIODS AVAILABLE
-----------------
  2015_2021 / 2018_2022 / 2019_2023 (latest, recommended)

OUTPUT
------
  <out-dir>/
    EGMS_L2b_<track>_<burst>_<pass>_<period>.zip   raw downloaded zips
    EGMS_L2b_<track>_<burst>_<pass>_<period>.csv   extracted CSVs (if --extract)
    egmslist.pkl                                    tile search cache (reusable)
    download_manifest.json                          list of tiles found
"""

import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SEP  = "=" * 68
SEP2 = "-" * 68

VALID_PRODUCTS = ["L2a", "L2b"]
VALID_PERIODS  = ["2015_2021", "2018_2022", "2019_2023"]


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for mod in ["EGMStoolkit", "fiona", "rasterio"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(msg, default=None, choices=None):
    prompt_str = f"  {msg} [{default}]: " if default else f"  {msg}: "
    while True:
        val = input(prompt_str).strip()
        if not val and default is not None:
            return default
        if not val:
            print("    Required — please enter a value.")
            continue
        if choices and val.lower() not in [c.lower() for c in choices]:
            print(f"    Must be one of: {', '.join(choices)}")
            continue
        return val


def _prompt_bbox():
    print("\n  Enter bounding box in WGS84 decimal degrees (lon/lat):")
    print("  Example Po Valley:  lon_min=6.55  lat_min=43.63  lon_max=14.02  lat_max=47.18")
    while True:
        try:
            lon_min = float(_prompt("lon_min (west,  °E)"))
            lat_min = float(_prompt("lat_min (south, °N)"))
            lon_max = float(_prompt("lon_max (east,  °E)"))
            lat_max = float(_prompt("lat_max (north, °N)"))
            if not (-180 <= lon_min < lon_max <= 180):
                print("    lon_min must be < lon_max, both in [-180, 180]")
                continue
            if not (-90 <= lat_min < lat_max <= 90):
                print("    lat_min must be < lat_max, both in [-90, 90]")
                continue
            area = (lon_max - lon_min) * (lat_max - lat_min)
            if area > 100:
                print(f"    Large area ({area:.1f} deg²) — this may be many tiles (~{int(area*8)} files).")
                if input("    Continue? [y/N]: ").strip().lower() != "y":
                    continue
            return lon_min, lat_min, lon_max, lat_max
        except ValueError:
            print("    Please enter numeric values.")


# ─────────────────────────────────────────────────────────────────────────────
# CORE DOWNLOAD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _write_bbox_shapefile(bbox, work_dir: Path) -> str:
    """
    Write bbox as a MultiLineString shapefile at work_dir/bbox.shp.
    Replicates exactly what EGMStoolkit createROI() does when bbox is a list,
    bypassing the GMT dependency (which is only used for country-name mode).
    Returns the path string to bbox.shp.
    """
    import fiona
    from fiona.crs import from_epsg
    from shapely.geometry import MultiLineString, mapping

    lon_min, lat_min, lon_max, lat_max = bbox

    # EGMStoolkit writes the bbox outline as a MultiLineString (closed ring)
    ring = [(lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
            (lon_min, lat_min)]
    multi = MultiLineString([ring])

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(work_dir / "bbox.shp")

    # Remove any existing bbox.shp files first (as createROI does)
    for ext in ["cpg", "dbf", "prj", "shp", "shx"]:
        p = work_dir / f"bbox.{ext}"
        if p.exists():
            p.unlink()

    schema = {"geometry": "MultiLineString", "properties": {"FID": "int"}}
    with fiona.open(out_path, mode="w", driver="ESRI Shapefile",
                    schema=schema, crs="EPSG:4326") as dst:
        dst.write({"geometry": mapping(multi), "properties": {"FID": 1}})

    return out_path


def run_download(
    token:    str,
    bbox:     tuple,        # (lon_min, lat_min, lon_max, lat_max)
    products: list,
    period:   str,
    out_dir:  Path,
    extract:  bool = True,
    clean_zips: bool = False,
    dry_run:  bool = False,
):
    from EGMStoolkit.classes.EGMSS1burstIDapi import S1burstIDmap
    from EGMStoolkit.classes.EGMSS1ROIapi    import S1ROIparameter
    from EGMStoolkit.classes.EGMSdownloaderapi import egmsdownloader

    out_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox

    # EGMStoolkit expects bbox as "lon_min,lat_min,lon_max,lat_max" string
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    # EGMStoolkit's 3rdparty dir (contains the S1 burst ID map files)
    egms_src = Path(__file__).parent / "EGMStoolkit" / "src" / "EGMStoolkit"
    dirmap   = str(egms_src / "3rdparty")

    print(f"\n{SEP}")
    print("  EGMS DOWNLOAD")
    print(f"{SEP}")
    print(f"  Bounding box : {lon_min}°E – {lon_max}°E, {lat_min}°N – {lat_max}°N")
    print(f"  Products     : {', '.join(products)}")
    print(f"  Period       : {period}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Extract CSVs : {'YES' if extract else 'NO'}")
    print(f"  Token        : {token[:8]}...{token[-4:]}")
    if dry_run:
        print(f"  *** DRY RUN — no files will be downloaded ***")

    # ── Step 1: Load / download S1 burst ID map ───────────────────────────
    print(f"\n{SEP2}\n  Step 1/3: Loading Sentinel-1 burst ID map ...\n{SEP2}",
          flush=True)
    burst_id = S1burstIDmap(dirmap=dirmap, verbose=True)
    burst_id.checkfile()
    # Download burst map if not present locally
    if not burst_id.list_date:
        print("  Burst ID map not found locally — downloading ...", flush=True)
        burst_id.downloadfile()
        burst_id.checkfile()

    all_tiles = []
    manifest  = {}

    # Build ROI shapefile from bbox manually — bypasses GMT dependency
    # createROI() with a list bbox writes a MultiLineString to workdir/bbox.shp
    # and sets self.ROIs to that path. We replicate this exactly without GMT.
    roi_shp = _write_bbox_shapefile(bbox, out_dir)
    print(f"  ROI shapefile written: {roi_shp}", flush=True)

    for product in products:
        print(f"\n{SEP2}\n  Step 2/3: Finding {product} tiles for bbox ...\n{SEP2}",
              flush=True)

        roi = S1ROIparameter(
            bbox         = bbox_str,
            egmslevel    = product,
            release      = period,
            workdirectory= str(out_dir),
            verbose      = True,
        )

        # Inject ROI path directly — bypasses createROI() / GMT
        # self.ROIs must be the path string to workdir/bbox.shp
        roi.ROIs = roi_shp

        roi.detectfromIDmap(burst_id)

        # Collect tile info from Data dict
        n_tiles = 0
        if hasattr(roi, "Data") and roi.Data:
            for track, track_data in roi.Data.items():
                for pass_dir, pass_data in track_data.items():
                    if isinstance(pass_data, dict):
                        for burst, burst_data in pass_data.items():
                            n_tiles += 1
        print(f"  Found tiles for {product}: checking ...", flush=True)

        # Save tile list cache
        cache_path = str(out_dir / f"egmslist_{product}_{period}.pkl")
        roi.saveIDlistL2(output=cache_path)
        print(f"  Tile cache saved: {cache_path}", flush=True)

        manifest[product] = {
            "bbox": bbox_str,
            "period": period,
            "cache": cache_path,
        }

        if dry_run:
            print(f"\n  DRY RUN: tile search complete for {product}.")
            print(f"  Inspect cache file to see tile list: {cache_path}")
            roi.printlist() if hasattr(roi, "printlist") else None
            continue

        # ── Step 3: Download ──────────────────────────────────────────────
        print(f"\n{SEP2}\n  Step 3/3: Downloading {product} tiles ...\n{SEP2}",
              flush=True)

        downloader = egmsdownloader(token=token, verbose=True)
        downloader.updatelist(roi)
        downloader.printlist()

        t0 = time.time()
        # Note: EGMStoolkit downloads to outputdir/L2b/2019_2023/ subdirectory
        # force=True is required due to a bug in EGMStoolkit where force=False
        # always triggers "Already downloaded" due to a missing os.path.isfile()
        # check. Resume is handled by checking for existing ZIP files instead.
        dl_dir = str(out_dir)
        downloader.download(
            outputdir  = dl_dir,
            unzipmode  = extract,
            cleanmode  = clean_zips,
            force      = True,
            verbose    = True,
        )
        # Tell user where files actually landed
        import pathlib
        actual_dir = out_dir / product / period
        if actual_dir.exists():
            n_zips = len(list(actual_dir.glob("*.zip")))
            n_csvs = len(list(actual_dir.glob("*.csv")))
            print(f"  Files in {actual_dir}: {n_zips} ZIPs, {n_csvs} CSVs",
                  flush=True)
        elapsed = time.time() - t0
        print(f"\n  {product} download complete in {elapsed:.0f}s", flush=True)

    # Save manifest
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest saved: {manifest_path}", flush=True)

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download EGMS InSAR tiles by bounding box (WGS84 lat/lon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--token",   help="EGMS token (from ?id= in download URL)")
    parser.add_argument("--bbox",    nargs=4, type=float,
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="Bounding box in WGS84 decimal degrees")
    parser.add_argument("--product", nargs="+", default=None,
                        choices=VALID_PRODUCTS,
                        help="Product level(s): L2a L2b (default: L2b)")
    parser.add_argument("--period",  default=None,
                        choices=VALID_PERIODS,
                        help="Time period (default: 2019_2023)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: ./egms_raw)")
    parser.add_argument("--no-extract",  action="store_true",
                        help="Keep ZIP files, do not extract CSVs")
    parser.add_argument("--clean-zips",  action="store_true",
                        help="Delete ZIP files after extraction")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Find tiles and save cache without downloading")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EGMS DATA DOWNLOADER")
    print(f"{SEP}")

    # ── Check dependencies ────────────────────────────────────────────────
    missing = _check_deps()
    if missing:
        print(f"  Missing dependencies: {', '.join(missing)}")
        print("  Install with:")
        for m in missing:
            if m == "EGMStoolkit":
                print("    git clone https://github.com/alexisInSAR/EGMStoolkit.git")
                print("    (add to site-packages .pth file)")
            else:
                print(f"    conda install -c conda-forge {m}")
        sys.exit(1)
    print("  Dependencies OK", flush=True)

    # ── Collect parameters ────────────────────────────────────────────────
    interactive = not all([args.token, args.bbox, args.period])
    if interactive:
        print("\n  Interactive mode — press Enter to accept defaults.\n")

    token = args.token or _prompt(
        "Token (from ?id= in EGMS download URL)"
    )

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = _prompt_bbox()

    products = args.product or (
        [_prompt("Product (L2a/L2b)", default="L2b",
                 choices=VALID_PRODUCTS).upper()]
        if interactive else ["L2b"]
    )

    period = args.period or _prompt(
        "Period", default="2019_2023", choices=VALID_PERIODS
    )

    out_dir = Path(args.out_dir or (
        _prompt("Output directory", default="./egms_raw")
        if interactive else "./egms_raw"
    ))

    # ── Run ───────────────────────────────────────────────────────────────
    run_download(
        token     = token,
        bbox      = bbox,
        products  = products,
        period    = period,
        out_dir   = out_dir,
        extract   = not args.no_extract,
        clean_zips= args.clean_zips,
        dry_run   = args.dry_run,
    )

    print(f"\n{SEP}")
    if not args.dry_run:
        print(f"  Done. CSV files are in: {out_dir}")
        print(f"  Next step:")
        print(f"    python egms_to_geoparquet.py --input-dir {out_dir}")
    else:
        print(f"  Dry run complete. Re-run without --dry-run to download.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
