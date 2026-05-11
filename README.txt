EGMSpy InSAR time series management and analysis Python pipeline - by Filippo Catani 
Machine Intelligence and Slope Stability Lab (MISSLab)  
University of Padova · Department of Geosciences ·
Version 2026-04

NOTE: for a more detailed reference on methods, algorithms, usage, and results on the demonstrator case, please refer to the following publication:

Catani F., Palmieri C., Todde R., Nava L., Floris M., 2026. EGMSpy: an open-source Python toolkit for scalable data handling, classification, clustering, and visualisation of Copernicus EGMS InSAR data. Submitted to EarthXiv.

---
What this is: 
A Python toolkit for processing, classifying, clustering and visualising Copernicus EGMS InSAR ground motion datasets at regional to continental scale (tested up to 300M points, designed to scale to >1B).
---
Pipeline scripts and their purpose 

`egms_download.py`	Download EGMS L2b tiles by bounding box via EGMStoolkit (Hrysiewicz et al., 2024)
`egms_to_geoparquet.py`	Convert EGMS CSV/ZIP → split GeoParquet (meta + TS pairs)
`egms_classify.py`	Physics-informed time series classification (rule + GMM)
`egms_cluster.py`	Velocity-weighted 3D DBSCAN spatial clustering
`egms_subcluster.py`	Per-class DBSCAN sub-clustering within parent clusters
`bridge.py`	Flask + DuckDB API server for viewer
`viewer.html`	Leaflet.js web map with single-canvas renderer
---

Environment setup (suggested)

conda create -n egms_tool python=3.12
conda activate egms_tool
conda install -c conda-forge pandas geopandas pyarrow duckdb scikit-learn \
              scipy shapely tqdm pyqt6 flask flask-cors fiona rasterio gdal
python -m pip install alive-progress plotly kaleido selenium concave-hull

# EGMStoolkit (for downloading only) (Hrysiewicz et al., 2024)
git clone https://github.com/alexisInSAR/EGMStoolkit.git
echo PATH_TO/EGMStoolkit/src > CONDA_ENV/Lib/site-packages/egmstoolkit.pth
---

Quick start (parameters' values are suggested based on the North Italy demonstrator case - Catani et al. 2026, EarthXiv)

1. Download EGMS data
---
python egms_download.py \
    --token YOUR_TOKEN \
    --bbox 6.55 43.63 14.02 47.18 \
    --product L2b --period 2019_2023 \
    --out-dir ./raw_data --no-extract

Get token from https://egms.land.copernicus.eu/ → any download link → `?id=TOKEN`

2. Convert to GeoParquet

python egms_to_geoparquet.py \
    --input-dir ./raw_data/L2b/2019_2023 \
    --out-dir ./processed_data
---

3. Classify time series
---
python egms_classify.py \
    --data-dir ./processed_data \
    --n-jobs 32
---
4. Cluster
---
python egms_cluster.py \
    --data-dir ./processed_data \
    --min-vel 2.5 --min-coh 0.5 \
    --eps-m 100 --eps-vel 3 \
    --min-samples 5 --buffer-m 20
---
5. Sub-cluster (optional)
---
python egms_subcluster.py \
    --data-dir ./processed_data \
    --eps-m 80 --eps-vel 3 \
    --min-samples 3 --n-jobs 32
---
6. Visualise
---
python egms_pipeline_gui.py
# or directly:
python bridge.py --data-dir ./processed_data
# then open viewer.html in browser

---
Performance notes on a multiprocessor machine (specs available upon request)
Step	NE Italy (300M points and time series)
Convert	~30 min	~8–12 hours
Classify	~5 min	~24–48 hours (32 workers)
Cluster (DBSCAN)	~90 sec	~15–30 min
Sub-cluster	~90 sec	~2–4 hours (32 workers)
Write-back	~25 sec	~10–20 min (16 parallel writers)
---
File structure
---
https://github.com/fcatani/EGMSpy/               ← all scripts live here
  egms_pipeline_gui.py
  egms_to_geoparquet.py
  egms_classify.py
  egms_cluster.py
  egms_subcluster.py        
  bridge.py
  viewer.html
  EGMStoolkit/              ← cloned from GitHub, for download only (Hrysiewicz et al., 2024)

---
Citation
If you use EGMSpy or any component of it, please cite:

1. Catani et al. (2026) "EGMSpy: an open-source Python toolkit for scalable data handling, classification, clustering, and visualisation of Copernicus EGMS InSAR data", EarthXiv, May 2026, DOI:10.31223/X55B59

2. Copernicus EGMS data: https://egms.land.copernicus.eu/

If you make use of the `egms_download.py`, also cite:

3. EGMStoolkit: Hrysiewicz et al. (2024), Earth Science Informatics, https://doi.org/10.1007/s12145-024-01356-w



   
