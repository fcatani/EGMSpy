#!/usr/bin/env python3
"""
EGMS Time Series Classifier  (v3 — revised class scheme)
MIT License

Copyright (c) 2026 Filippo Catani and MISS-Lab at UNIPD
=========================================================
Physics-informed feature extraction + hybrid rule/GMM classification
of EGMS InSAR time series.

Classification approach
-----------------------
  STAGE 1 — Rule-based (primary, deterministic):
    Physics thresholds applied to extracted features assign every eligible
    point to a class.  These labels are FINAL — no statistical model can
    override a rule-based decision.

  STAGE 2 — GMM (secondary, for ambiguous residual only):
    Points left unresolved by rules (label still -1) are passed to a GMM
    fitted on the full feature space.  GMM resolves genuinely ambiguous
    points; it cannot reassign any rule-claimed point.

Changes from v2
---------------
  • CLASS SCHEME simplified: jump (was class 6) and variable (was class 5)
    are NO LONGER primary classes.  This eliminates the dominant
    misclassification seen in v2 where series with a clear trend (linear /
    accel / decel) were overridden by the jump rule.

  • NEW DESCRIPTOR FLAGS (boolean, written to meta alongside periodic):
      jumpy       — ≥1 non-seasonal step ≥ jump_mm detected in the series
      variable    — high CUSUM score (irregular rate, erratic signal)
      noisy_trend — moving point (|v|≥v_stable) with high residual scatter

  • variable is kept as a FALLBACK CLASS (4) for points whose velocity is
    below the stable threshold yet shows high irregularity — distinguishing
    them from the clean stable/noisy classes.  For moving points, variable
    is only a descriptor flag; the dominant trend class is assigned instead.

  • accel / decel rules RELAXED: either strong split-half velocity difference
    OR good quadratic R² is sufficient (v2 required both).  This recovers
    many visually obvious accelerating series that slipped into linear.

  • periodic R² threshold lowered to 0.15 (was 0.25) for better recall of
    moderate seasonal signals.  Tunable via --seasonal-r2.

  • GMM n_components reduced to 5 (matching the 5 primary classes) —
    faster convergence, less memory.

Classification scheme (v3)
--------------------------
  0  stable       |v| < v_stable, low residual noise
  1  noisy        |v| < v_stable, high residual noise
  2  linear       |v| >= v_stable, sustained linear trend
  3  accel        accelerating deformation
  4  decel        decelerating deformation
  5  variable     low velocity but irregular / erratic (fallback stable class)
  6  other        residual — resolved by GMM

Supplementary descriptor flags (all boolean, independent of primary class)
---------------------------------------------------------------------------
  periodic     — significant annual/semi-annual harmonic component
  jumpy        — ≥1 non-seasonal step ≥ jump_mm in the series
  variable     — high CUSUM score / erratic rate changes
  noisy_trend  — moving point with high residual scatter after detrend+deseason

Output columns written back to *_meta.parquet (overwrite if present):
  class_1        (str)     primary class name
  class_2        (str)     second-most-likely class name
  class_prob_1   (float32) confidence: 1.0 for rule-assigned, GMM prob for others
  periodic       (bool)
  jumpy          (bool)
  variable_flag  (bool)    named variable_flag to avoid collision with class 5 name
  noisy_trend    (bool)

Usage
-----
  python egms_classify.py --data-dir ./processed_data
  python egms_classify.py --data-dir ./processed_data \\
      --min-coh 0.5 --min-vel 1.0 --v-stable 2.5 --jump-mm 5.0 \\
      --seasonal-r2 0.15 --chunk-size 50000

Requirements: pandas, numpy, pyarrow, duckdb, scikit-learn, scipy, tqdm
"""

import argparse
import re
import os
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS REGISTRY  (v3 — 6 primary classes + other)
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {
    0: "stable",
    1: "noisy",
    2: "linear",
    3: "accel",
    4: "decel",
    5: "variable",   # fallback for erratic low-velocity points
    6: "other",      # GMM-resolved residual
}
N_CLASSES = len(CLASS_NAMES)

# ══════════════════════════════════════════════════════════════════════════════
# DATE COLUMN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DATE_PAT = re.compile(r'^D?(\d{8})$')

def _is_date_col(c) -> bool:
    return bool(_DATE_PAT.match(str(c).strip()))

def _bare(c) -> str:
    m = _DATE_PAT.match(str(c).strip())
    return m.group(1) if m else str(c)

def _to_decimal_year(date_str: str) -> float:
    import datetime
    d = datetime.datetime.strptime(date_str, "%Y%m%d")
    year_start = datetime.datetime(d.year, 1, 1)
    year_end   = datetime.datetime(d.year + 1, 1, 1)
    frac = (d - year_start).total_seconds() / (year_end - year_start).total_seconds()
    return d.year + frac

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (vectorised per-chunk)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_chunk(
    ts_array:    np.ndarray,   # (n_points, n_dates)  float32, NaN for gaps
    t_years:     np.ndarray,   # (n_dates,)  fractional year
    date_months: np.ndarray,   # (n_dates,)  integer month 1-12
    winter_mask: np.ndarray,   # (n_dates,)  bool, True = Nov-Mar
    jump_thresh_mm: float,
    v_stable_max:   float,
    seasonal_r2_thresh: float,
) -> np.ndarray:
    """
    Extract 18 features per point.  Fully vectorised NumPy — no Python loop
    over points (except for jump/CUSUM/ACF which require per-point logic).
    Returns shape (n_points, 18).

    Features (column index):
      0  v_robust          OLS slope (mm/yr)
      1  v_first_half      mean velocity first 50% of time span
      2  v_second_half     mean velocity second 50% of time span
      3  v_diff            v_second_half - v_first_half
      4  rmse_detrended    RMSE after linear detrend
      5  rmse_deseason     RMSE after detrend + seasonal removal
      6  seasonal_amp      amplitude of annual harmonic (mm)
      7  seasonal_r2       R² of harmonic fit on detrended series
      8  quad_coeff        t² coefficient (acceleration proxy, mm/yr²)
      9  quad_r2           R² improvement of quadratic over linear
     10  n_jumps_nonseas   count of non-seasonal jumps
     11  jump_max_mm       largest jump magnitude (mm)
     12  n_valid_frac      fraction of non-null samples
     13  acf_lag1          lag-1 autocorrelation of detrended residuals
     14  cusum_score       CUSUM changepoint score (normalised)
     15  velocity_r2       R² of linear fit
     16  range_mm          total range max-min
     17  noise_p90         90th pct of |first differences|
    """
    n_pts, n_dates = ts_array.shape
    feats = np.full((n_pts, 18), np.nan, dtype=np.float32)

    t      = t_years - t_years.mean()
    half   = n_dates // 2
    first_mask  = np.arange(n_dates) < half
    second_mask = ~first_mask

    omega = 2 * np.pi
    H = np.column_stack([
        np.ones(n_dates),
        t,
        np.cos(omega * t_years), np.sin(omega * t_years),
        np.cos(2 * omega * t_years), np.sin(2 * omega * t_years),
    ])
    L = np.column_stack([np.ones(n_dates), t])
    Q = np.column_stack([np.ones(n_dates), t, t**2])

    valid_mask = ~np.isnan(ts_array)
    n_valid    = valid_mask.sum(axis=1)
    feats[:, 12] = n_valid / n_dates

    ok = n_valid >= 10
    if not ok.any():
        return feats

    def _ols_batch(X, Y, mask):
        Y_m  = np.where(mask, Y, 0.0)
        k    = X.shape[1]
        Xb   = X[np.newaxis, :, :]
        maskb = mask[:, :, np.newaxis]
        mX   = Xb * maskb
        XtX  = np.einsum('pdk,pdl->pkl', mX, mX)
        XtY  = np.einsum('pdk,pd->pk',   mX, Y_m)
        coeffs = np.zeros((Y.shape[0], k), dtype=np.float64)
        for i in np.where(ok)[0]:
            try:
                coeffs[i] = np.linalg.solve(XtX[i], XtY[i])
            except np.linalg.LinAlgError:
                coeffs[i] = np.linalg.lstsq(XtX[i], XtY[i], rcond=None)[0]
        pred  = (X[np.newaxis, :, :] * coeffs[:, np.newaxis, :]).sum(axis=2)
        resid = Y - pred
        return coeffs.astype(np.float32), resid.astype(np.float32)

    # ── Linear fit ────────────────────────────────────────────────────────
    L_coeffs, L_resid = _ols_batch(L, ts_array, valid_mask)
    slope = L_coeffs[:, 1]
    feats[:, 0] = slope

    y_mean  = np.nanmean(ts_array, axis=1)
    ss_tot  = np.nansum((ts_array - y_mean[:, None])**2, axis=1).clip(min=1e-9)
    ss_res_L = np.nansum((L_resid * valid_mask)**2, axis=1)
    feats[:, 15] = (1 - ss_res_L / ss_tot).clip(-1, 1)
    feats[:, 4]  = np.sqrt(np.nanmean(np.where(valid_mask, L_resid**2, np.nan), axis=1))
    feats[:, 16] = np.nanmax(ts_array, axis=1) - np.nanmin(ts_array, axis=1)
    feats[~ok, 4]  = np.nan
    feats[~ok, 16] = np.nan

    # ── Split-half velocities ─────────────────────────────────────────────
    mask1 = valid_mask & first_mask[np.newaxis, :]
    mask2 = valid_mask & second_mask[np.newaxis, :]
    ok12  = ok & (mask1.sum(axis=1) >= 5) & (mask2.sum(axis=1) >= 5)
    if ok12.any():
        sub = np.where(ok12)[0]
        L1c, _ = _ols_batch(L, ts_array[sub], mask1[sub])
        L2c, _ = _ols_batch(L, ts_array[sub], mask2[sub])
        feats[sub, 1] = L1c[:, 1]
        feats[sub, 2] = L2c[:, 1]
        feats[sub, 3] = L2c[:, 1] - L1c[:, 1]

    # ── Harmonic (seasonal) fit ───────────────────────────────────────────
    H_coeffs, H_resid = _ols_batch(H, ts_array, valid_mask)
    trend_pred = H_coeffs[:, 0:1] + H_coeffs[:, 1:2] * t[np.newaxis, :]
    harm_pred  = (H[np.newaxis] * H_coeffs[:, np.newaxis, :]).sum(axis=2)
    seasonal   = harm_pred - trend_pred
    deseason   = L_resid - seasonal

    annual_amp = np.sqrt(H_coeffs[:, 2]**2 + H_coeffs[:, 3]**2)
    feats[:, 6] = annual_amp.astype(np.float32)

    L_resid_v  = np.where(valid_mask, L_resid, np.nan)
    deseason_v = np.where(valid_mask, deseason, np.nan)
    ss_detrended = np.nansum(L_resid_v**2,  axis=1).clip(min=1e-9)
    ss_deseason  = np.nansum(deseason_v**2, axis=1)
    feats[:, 7] = (1 - ss_deseason / ss_detrended).clip(-1, 1).astype(np.float32)
    feats[:, 5] = np.sqrt(np.nanmean(deseason_v**2, axis=1)).astype(np.float32)
    feats[~ok, 5] = np.nan
    feats[~ok, 6] = np.nan
    feats[~ok, 7] = np.nan

    # ── Quadratic fit (acceleration) ─────────────────────────────────────
    Q_coeffs, Q_resid = _ols_batch(Q, ts_array, valid_mask)
    feats[:, 8] = Q_coeffs[:, 2].astype(np.float32)
    ss_res_Q    = np.nansum((Q_resid * valid_mask)**2, axis=1)
    feats[:, 9] = ((ss_res_L - ss_res_Q) / ss_tot).clip(-1, 1).astype(np.float32)

    # ── Jump detection (per-point loop — unavoidable) ─────────────────────
    for i in np.where(ok)[0]:
        y   = ts_array[i]
        vm  = valid_mask[i]
        idx = np.where(vm)[0]
        if len(idx) < 4:
            continue
        dt_pair = np.maximum(
            t_years[idx[1:]] * 365.25 - t_years[idx[:-1]] * 365.25, 1.0
        )
        deseas_i  = deseason[i]
        diffs_ds  = np.diff(deseas_i[idx])
        rates_ds  = np.abs(diffs_ds) / dt_pair
        q75       = np.percentile(rates_ds, 75)
        thresh_r  = max(q75 * 4, jump_thresh_mm / 6.0)
        jmask     = rates_ds > thresh_r
        feats[i, 10] = float(jmask.sum())
        feats[i, 11] = float(np.abs(diffs_ds[jmask]).max()) if jmask.any() else 0.0

    # ── Lag-1 ACF of detrended residuals ─────────────────────────────────
    for i in np.where(ok)[0]:
        r = L_resid[i][valid_mask[i]]
        if len(r) >= 4:
            r_c   = r - r.mean()
            denom = (r_c**2).sum()
            if denom > 1e-9:
                feats[i, 13] = float((r_c[:-1] * r_c[1:]).sum() / denom)

    # ── CUSUM changepoint score ───────────────────────────────────────────
    for i in np.where(ok)[0]:
        r = L_resid[i][valid_mask[i]]
        if len(r) >= 6:
            r_std = r.std()
            if r_std > 1e-9:
                r_n   = (r - r.mean()) / r_std
                cusum = np.cumsum(r_n)
                feats[i, 14] = float((cusum.max() - cusum.min()) / np.sqrt(len(r)))

    # ── noise_p90 ─────────────────────────────────────────────────────────
    for i in np.where(ok)[0]:
        r = ts_array[i][valid_mask[i]]
        if len(r) >= 4:
            feats[i, 17] = float(np.percentile(np.abs(np.diff(r)), 90))

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED PRIMARY LABELLING  (v3)
# ══════════════════════════════════════════════════════════════════════════════

def rule_based_labels(
    feats:          np.ndarray,  # (n_pts, 18)
    v_stable_max:   float,
    noise_thresh:   float,
    quad_r2_thresh: float,
    accel_thresh:   float,
    cusum_thresh:   float,
    noisy_trend_thresh: float,   # RMSE deseason threshold for noisy_trend flag
) -> np.ndarray:
    """
    Rule-based primary classifier.  Returns integer class indices
    (0-6) or -1 (uncertain → GMM).

    Priority order (top = highest):
      1. stable / noisy      (velocity gate, decisive)
      2. accel / decel       (RELAXED: strong v_diff OR good quad_r2 sufficient)
      3. linear              (good linear R², no accel/decel signal)
      4. variable            (erratic low-velocity — fallback stable group)
      5. -1 (other)          → GMM

    NOTE: jump is no longer a class.  It is computed as a separate boolean
    descriptor flag (jumpy) after classification.
    """
    n = feats.shape[0]
    labels = np.full(n, -1, dtype=np.int8)

    v       = feats[:, 0]
    v_diff  = feats[:, 3]
    rmse_dt = feats[:, 4]
    q_coeff = feats[:, 8]
    q_r2    = feats[:, 9]
    cusum   = feats[:, 14]
    lin_r2  = feats[:, 15]

    abs_v         = np.abs(v)
    is_stable_vel = abs_v < v_stable_max
    valid_pts     = ~np.isnan(v)

    # ── Class 0: Stable ───────────────────────────────────────────────────
    labels[valid_pts & is_stable_vel & (rmse_dt <= noise_thresh)] = 0

    # ── Class 1: Noisy stable ─────────────────────────────────────────────
    labels[valid_pts & is_stable_vel & (rmse_dt > noise_thresh)] = 1

    # ── Class 5: Variable (erratic low-velocity, NOT a moving class) ──────
    # Applied here (within stable-velocity zone) so it complements stable/noisy.
    # For moving points, variable is only a flag — NOT overriding a trend class.
    mask_var_stable = (
        valid_pts & is_stable_vel
        & (labels == -1)          # shouldn't happen after 0/1, but safety guard
        & (cusum > cusum_thresh)
        & (lin_r2 < 0.5)
    )
    labels[mask_var_stable] = 5

    # ── Class 3: Accelerating ─────────────────────────────────────────────
    # Convention: v is LOS (negative = away from satellite / subsidence).
    # Accelerating = magnitude growing over time:
    #   subsiding series (v < 0): v_diff must also be negative (more negative
    #     second half) AND quadratic curvature q_coeff must be negative.
    #   uplifting series (v > 0): v_diff positive AND q_coeff positive.
    # Require BOTH quad_r2 improvement AND v_diff to avoid false positives on
    # nearly-linear series with minor noise asymmetry.
    # Additional guard: lin_r2 < 0.85 — a highly linear series is not accel.
    sign_accel = ((v < 0) & (v_diff < -accel_thresh) & (q_coeff < 0)) \
               | ((v > 0) & (v_diff >  accel_thresh) & (q_coeff > 0))
    mask_accel = (
        valid_pts & ~is_stable_vel
        & (labels == -1)
        & sign_accel
        & (q_r2 > quad_r2_thresh)
        & (lin_r2 < 0.85)
    )
    labels[mask_accel] = 3

    # ── Class 4: Decelerating ─────────────────────────────────────────────
    # Decelerating = magnitude shrinking:
    #   subsiding series (v < 0): v_diff positive (less negative second half)
    #     AND q_coeff positive (concave up).
    #   uplifting series (v > 0): v_diff negative AND q_coeff negative.
    # Same guards as accel: require quad_r2 AND not a clean linear series.
    sign_decel = ((v < 0) & (v_diff >  accel_thresh) & (q_coeff > 0)) \
               | ((v > 0) & (v_diff < -accel_thresh) & (q_coeff < 0))
    mask_decel = (
        valid_pts & ~is_stable_vel
        & (labels == -1)
        & sign_decel
        & (q_r2 > quad_r2_thresh)
        & (lin_r2 < 0.85)
    )
    labels[mask_decel] = 4

    # ── Class 2: Linear ───────────────────────────────────────────────────
    mask_lin = (
        valid_pts & ~is_stable_vel
        & (labels == -1)
        & (lin_r2 >= 0.65)
    )
    labels[mask_lin] = 2

    # ── Remaining unresolved moving points → -1 (GMM) ────────────────────
    # No explicit variable override here: for moving points, variable is a flag.

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTOR FLAGS  (vectorised, independent of primary class)
# ══════════════════════════════════════════════════════════════════════════════

def compute_descriptor_flags(
    feats:              np.ndarray,
    v_stable_max:       float,
    noise_thresh:       float,
    cusum_thresh:       float,
    jump_min_mm:        float,
    seasonal_r2_thresh: float,
    noisy_trend_thresh: float,
) -> dict:
    """
    Compute four boolean descriptor arrays, each shape (n_pts,).

    periodic     — significant annual/semi-annual harmonic
    jumpy        — ≥1 non-seasonal step ≥ jump_min_mm
    variable_flag — high CUSUM score (erratic rate)
    noisy_trend  — moving point with high residual scatter
    """
    abs_v         = np.abs(feats[:, 0])
    is_moving     = abs_v >= v_stable_max
    valid_pts     = ~np.isnan(feats[:, 0])

    periodic = feats[:, 7] > seasonal_r2_thresh

    jumpy = (feats[:, 10] >= 1) & (feats[:, 11] > jump_min_mm)

    # variable_flag: applicable to both stable and moving points
    variable_flag = (feats[:, 14] > cusum_thresh) & valid_pts

    # noisy_trend: moving but scattered after full detrend+deseason
    noisy_trend = (
        valid_pts & is_moving
        & (feats[:, 5] > noisy_trend_thresh)   # rmse_deseason
    )

    return {
        "periodic":      periodic.astype(bool),
        "jumpy":         jumpy.astype(bool),
        "variable_flag": variable_flag.astype(bool),
        "noisy_trend":   noisy_trend.astype(bool),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GMM CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def fit_gmm(
    feats_scaled: np.ndarray,
    rule_labels:  np.ndarray,
    n_components: int,
    random_state: int,
) -> GaussianMixture:
    """
    Fit a GMM with semi-supervised initialisation seeded from rule-label centroids.
    """
    valid = ~np.isnan(feats_scaled).any(axis=1)
    X     = feats_scaled[valid].astype(np.float64)
    n_pts = len(X)
    n_feats = X.shape[1]

    max_safe     = max(2, n_pts // max(5 * n_feats, 20))
    n_components = min(n_components, max_safe)
    if n_components < 6:
        print(f"  ⚠  Auto-reduced GMM components to {n_components} "
              f"(only {n_pts} valid points for {n_feats} features)")

    rule_valid   = rule_labels[valid]
    means_init   = np.zeros((n_components, n_feats))
    rng = np.random.RandomState(random_state)
    for cls in range(n_components):
        mask = (rule_valid == cls)
        if mask.sum() >= 2:
            means_init[cls] = X[mask].mean(axis=0)
        else:
            means_init[cls] = X[rng.choice(n_pts)]

    for cov_type in ('full', 'diag', 'spherical'):
        for reg in (1e-3, 1e-2, 1e-1, 0.5):
            try:
                init_kw = {}
                init_params = 'kmeans'
                if cov_type == 'full' and n_components <= max_safe:
                    init_kw    = {'means_init': means_init}
                    init_params = 'k-means++'
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    max_iter=300,
                    n_init=5,
                    tol=1e-4,
                    reg_covar=reg,
                    random_state=random_state,
                    verbose=0,
                    **init_kw,
                )
                gmm.fit(X)
                if cov_type != 'full' or reg > 1e-3:
                    print(f"  ℹ  GMM fitted with cov_type='{cov_type}', reg_covar={reg}")
                return gmm
            except Exception:
                continue

    raise RuntimeError(
        "GMM fitting failed for all covariance types and regularisation values. "
        "Try --n-components 2 or reduce --min-coh."
    )


def _align_gmm_components(gmm, feats_scaled, rule_labels) -> dict:
    """Map each GMM component to the most frequent rule-based class."""
    valid = ~np.isnan(feats_scaled).any(axis=1)
    actual_n = gmm.n_components
    if valid.sum() == 0:
        return {i: i % N_CLASSES for i in range(actual_n)}

    gmm_labels = gmm.predict(feats_scaled[valid].astype(np.float64))
    rule_v     = rule_labels[valid]

    mapping = {}
    for comp in range(actual_n):
        mask = gmm_labels == comp
        if mask.sum() == 0:
            mapping[comp] = comp % N_CLASSES
            continue
        rl   = rule_v[mask]
        vals, counts = np.unique(rl[rl >= 0], return_counts=True)
        if len(vals) == 0:
            mapping[comp] = comp % N_CLASSES
        else:
            mapping[comp] = int(vals[counts.argmax()])
    return mapping


def gmm_predict(
    gmm:          GaussianMixture,
    feats_scaled: np.ndarray,
    rule_labels:  np.ndarray,
    class_names:  dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict primary and secondary class from GMM probabilities."""
    n       = len(feats_scaled)
    class_1 = np.empty(n, dtype=object)
    class_2 = np.empty(n, dtype=object)
    prob_1  = np.zeros(n, dtype=np.float32)

    valid = ~np.isnan(feats_scaled).any(axis=1)

    if valid.sum() > 0:
        probs = gmm.predict_proba(feats_scaled[valid].astype(np.float64))
        component_to_class = _align_gmm_components(gmm, feats_scaled, rule_labels)

        n_cls        = len(class_names)
        class_probs  = np.zeros((valid.sum(), n_cls), dtype=np.float32)
        for comp_idx, cls_idx in component_to_class.items():
            if comp_idx >= probs.shape[1]:
                continue
            cls_idx = min(cls_idx, n_cls - 1)
            class_probs[:, cls_idx] += probs[:, comp_idx]

        row_sums = class_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        class_probs /= row_sums

        sorted_idx = np.argsort(-class_probs, axis=1)
        vi = np.where(valid)[0]
        class_1[vi] = [class_names[sorted_idx[j, 0]] for j in range(len(vi))]
        class_2[vi] = [class_names[sorted_idx[j, 1]] for j in range(len(vi))]
        prob_1[vi]  = class_probs[np.arange(len(vi)), sorted_idx[:, 0]]

    invalid = ~valid
    if invalid.sum() > 0:
        fb = rule_labels[invalid]
        class_1[invalid] = [class_names.get(int(l), 'other') if l >= 0 else 'other'
                            for l in fb]
        class_2[invalid] = ['other'] * invalid.sum()
        prob_1[invalid]  = 0.0

    return class_1, class_2, prob_1


# ══════════════════════════════════════════════════════════════════════════════
# CLASS-2 FILL FOR RULE-ASSIGNED POINTS
# ══════════════════════════════════════════════════════════════════════════════

def _fill_class2_for_ruled(
    class_1:     np.ndarray,
    class_2:     np.ndarray,
    rule_labels: np.ndarray,
    feats:       np.ndarray,
    class_names: dict,
    other_mask:  np.ndarray,
) -> None:
    """
    For rule-assigned points, set class_2 to the nearest-centroid class
    (by Euclidean distance in feature space) that differs from class_1.
    Gives a meaningful second label in the viewer without a full GMM pass.
    """
    ruled_mask  = ~other_mask & (rule_labels >= 0)
    if ruled_mask.sum() == 0:
        return
    valid = ~np.isnan(feats).any(axis=1)

    # Build per-class centroids
    centroids = {}
    for cls_idx, cls_name in class_names.items():
        m = ruled_mask & valid & (rule_labels == cls_idx)
        if m.sum() >= 2:
            centroids[cls_idx] = feats[m].mean(axis=0)

    if len(centroids) < 2:
        return

    cls_indices  = list(centroids.keys())
    centroid_mat = np.vstack([centroids[c] for c in cls_indices])

    target_mask = ruled_mask & valid
    target_idx  = np.where(target_mask)[0]
    if len(target_idx) == 0:
        return

    X     = feats[target_idx]
    diffs = X[:, np.newaxis, :] - centroid_mat[np.newaxis, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))

    own_labels = rule_labels[target_idx]
    for li, (ti, own) in enumerate(zip(target_idx, own_labels)):
        row = dists[li].copy()
        for ki, ci in enumerate(cls_indices):
            if ci == own:
                row[ki] = np.inf
        best_ki = int(np.argmin(row))
        class_2[ti] = class_names[cls_indices[best_ki]]


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def update_meta_parquet(
    meta_path:     Path,
    pid_series:    pd.Series,
    class_1:       np.ndarray,
    class_2:       np.ndarray,
    prob_1:        np.ndarray,
    periodic:      np.ndarray,
    jumpy:         np.ndarray,
    variable_flag: np.ndarray,
    noisy_trend:   np.ndarray,
):
    """Merge classification columns into meta parquet (atomic overwrite)."""
    table = pq.read_table(meta_path)
    df    = table.to_pandas()

    cls_df = pd.DataFrame({
        'pid':           pid_series.values,
        'class_1':       class_1.astype(str),
        'class_2':       class_2.astype(str),
        'class_prob_1':  prob_1.astype(np.float32),
        'periodic':      periodic.astype(bool),
        'jumpy':         jumpy.astype(bool),
        'variable_flag': variable_flag.astype(bool),
        'noisy_trend':   noisy_trend.astype(bool),
    })

    drop_cols = ['class_1', 'class_2', 'class_prob_1',
                 'periodic', 'jumpy', 'variable_flag', 'noisy_trend']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.merge(cls_df, on='pid', how='left')

    tmp       = meta_path.with_suffix('.tmp.parquet')
    new_table = pa.Table.from_pandas(df, preserve_index=False)
    orig_meta = table.schema.metadata or {}
    new_table = new_table.replace_schema_metadata(orig_meta)
    pq.write_table(new_table, tmp, compression='snappy')
    tmp.replace(meta_path)


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "═" * 68
SEP2 = "─" * 68

def print_class_distribution(class_1: np.ndarray, prefix: str = ""):
    vals, counts = np.unique(class_1, return_counts=True)
    total = len(class_1)
    print(f"\n{prefix}  Class distribution (primary):")
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        bar = "█" * int(30 * c / total)
        print(f"{prefix}    {v:<14}  {c:>8,}  {100*c/total:5.1f}%  {bar}")


def print_flag_distribution(flags: dict, n_total: int, prefix: str = ""):
    print(f"\n{prefix}  Descriptor flags:")
    for name, arr in flags.items():
        n = int(arr.sum())
        pct = 100 * n / max(n_total, 1)
        print(f"{prefix}    {name:<16}  {n:>8,}  ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION (per parquet pair)
# ══════════════════════════════════════════════════════════════════════════════

def classify_pair(
    meta_path:          Path,
    ts_path:            Path,
    min_coh:            float,
    min_vel:            float,
    v_stable_max:       float,
    noise_thresh:       float,
    quad_r2_thresh:     float,
    accel_thresh:       float,
    cusum_thresh:       float,
    jump_min_mm:        float,
    seasonal_r2_thresh: float,
    noisy_trend_thresh: float,
    n_gmm_components:   int,
    chunk_size:         int,
    random_state:       int,
):
    t0 = time.time()
    print(f"\n  {SEP2}")
    print(f"  File: {meta_path.stem.replace('_meta','')}")
    print(f"  {SEP2}")

    # ── Load metadata ──────────────────────────────────────────────────────
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{meta_path.as_posix()}')")
    con.execute(f"CREATE VIEW ts   AS SELECT * FROM read_parquet('{ts_path.as_posix()}')")

    # ── Discover date columns ──────────────────────────────────────────────
    ts_schema = con.execute("DESCRIBE ts").df()
    all_cols  = ts_schema["column_name"].tolist()
    date_cols = sorted([c for c in all_cols if _is_date_col(c)], key=_bare)
    n_dates   = len(date_cols)

    if n_dates < 20:
        print(f"  ✗ Too few date columns ({n_dates}) — skipping")
        return

    t_years     = np.array([_to_decimal_year(_bare(c)) for c in date_cols])
    date_months = np.array([int(_bare(c)[4:6]) for c in date_cols], dtype=np.int8)
    winter_mask = np.isin(date_months, [11, 12, 1, 2, 3])

    print(f"  Date range   : {_bare(date_cols[0])} → {_bare(date_cols[-1])}")
    print(f"  Dates        : {n_dates}")

    # ── Coherence stats ────────────────────────────────────────────────────
    n_total   = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    coh_stats = con.execute("""
        SELECT MIN(temporal_coherence), MAX(temporal_coherence),
               AVG(temporal_coherence),
               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY temporal_coherence),
               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY temporal_coherence),
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY temporal_coherence)
        FROM meta
    """).fetchone()
    print(f"  Total points : {n_total:,}")
    print(f"  Coherence    : min={coh_stats[0]:.3f}  "
          f"p25={coh_stats[3]:.3f}  p50={coh_stats[4]:.3f}  "
          f"p75={coh_stats[5]:.3f}  max={coh_stats[1]:.3f}")

    coh_max = float(coh_stats[1])
    if min_coh > coh_max * 0.85:
        print(f"  ⚠  WARNING: min_coh={min_coh} is very close to dataset max "
              f"({coh_max:.3f}).  Suggested: --min-coh {round(float(coh_stats[3]),2)}")

    # ── Filter eligible points ─────────────────────────────────────────────
    eligible_pids = con.execute(f"""
        SELECT pid FROM meta
        WHERE temporal_coherence >= {min_coh}
          AND ABS(mean_velocity)  >= {min_vel}
    """).df()["pid"].tolist()
    n_eligible = len(eligible_pids)
    pct = 100 * n_eligible / max(n_total, 1)
    print(f"  Eligible     : {n_eligible:,}  ({pct:.1f}%)  "
          f"(coh≥{min_coh}, |vel|≥{min_vel})")

    if n_eligible == 0:
        print("  ✗ No eligible points — lower --min-coh threshold")
        return
    if pct < 1.0:
        print(f"  ⚠  Only {pct:.2f}% eligible — consider lowering --min-coh")

    # ── Feature extraction  (Strategy 3: single full-table read) ────────────
    # Read the ENTIRE ts parquet into memory once, filter to eligible PIDs,
    # then process in numpy chunks.  This eliminates thousands of SQL
    # round-trips (one per chunk) caused by the old WHERE pid IN (...) pattern,
    # which built a huge SQL string per chunk and re-scanned the parquet each time.
    print(f"  Loading full TS table into memory ...", flush=True)
    quoted_dates = ", ".join(f'"{c}"' for c in date_cols)
    ts_full = con.execute(f"SELECT pid, {quoted_dates} FROM ts").df()

    # Filter to eligible PIDs using a fast set lookup
    eligible_set = set(eligible_pids)
    mask_elig    = ts_full["pid"].isin(eligible_set)
    ts_elig      = ts_full[mask_elig].reset_index(drop=True)
    del ts_full   # free memory immediately
    print(f"  TS rows loaded : {len(ts_elig):,}", flush=True)

    all_pids  = []
    all_feats = []
    n_rows    = len(ts_elig)
    pid_col   = ts_elig["pid"].values
    ts_matrix = ts_elig[date_cols].values.astype(np.float32)
    ts_matrix[~np.isfinite(ts_matrix)] = np.nan
    del ts_elig   # free DataFrame, keep numpy arrays

    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    print(f"  Extracting features ({n_chunks} chunks of {chunk_size:,}) ...",
          flush=True)

    for ci in tqdm(range(n_chunks), desc="  Feature extract", unit="chunk",
                   leave=False,
                   bar_format="{desc} {percentage:3.0f}%|{bar:30}| "
                              "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        s = ci * chunk_size
        e = min(s + chunk_size, n_rows)
        pids    = pid_col[s:e]
        ts_vals = ts_matrix[s:e]
        feats   = extract_features_chunk(
            ts_vals, t_years, date_months, winter_mask,
            jump_thresh_mm=jump_min_mm,
            v_stable_max=v_stable_max,
            seasonal_r2_thresh=seasonal_r2_thresh,
        )
        all_pids.append(pids)
        all_feats.append(feats)

    del ts_matrix  # free large array

    if not all_feats:
        print("  ✗ No features extracted")
        return

    pids_arr  = np.concatenate(all_pids)
    feats_arr = np.vstack(all_feats)
    print(f"  Features shape: {feats_arr.shape}")

    # ── Rule-based primary labelling ───────────────────────────────────────
    print("  Rule-based classification (v3 — no jump class) ...")
    rule_labels = rule_based_labels(
        feats_arr,
        v_stable_max=v_stable_max,
        noise_thresh=noise_thresh,
        quad_r2_thresh=quad_r2_thresh,
        accel_thresh=accel_thresh,
        cusum_thresh=cusum_thresh,
        noisy_trend_thresh=noisy_trend_thresh,
    )
    rule_counts = {CLASS_NAMES[k]: int((rule_labels == k).sum())
                   for k in range(N_CLASSES)}
    print(f"  Rule labels  : {rule_counts}")

    # ── Descriptor flags (independent of primary class) ───────────────────
    print("  Computing descriptor flags ...")
    flags = compute_descriptor_flags(
        feats_arr,
        v_stable_max=v_stable_max,
        noise_thresh=noise_thresh,
        cusum_thresh=cusum_thresh,
        jump_min_mm=jump_min_mm,
        seasonal_r2_thresh=seasonal_r2_thresh,
        noisy_trend_thresh=noisy_trend_thresh,
    )

    # ── GMM on residual 'other' points ─────────────────────────────────────
    other_mask = (rule_labels == N_CLASSES - 1) | (rule_labels == -1)
    n_other    = int(other_mask.sum())
    n_ruled    = int((~other_mask).sum())
    print(f"  Rule-assigned  : {n_ruled:,}  ({100*n_ruled/max(len(rule_labels),1):.1f}%)")
    print(f"  GMM residual   : {n_other:,}  ({100*n_other/max(len(rule_labels),1):.1f}%)")

    n_pts   = len(rule_labels)
    class_1 = np.array([CLASS_NAMES.get(int(l), 'other') if l >= 0 else 'other'
                        for l in rule_labels], dtype=object)
    class_2 = np.array(['other'] * n_pts, dtype=object)
    prob_1  = np.ones(n_pts, dtype=np.float32)

    if n_other > 0:
        valid_mask   = ~np.isnan(feats_arr).any(axis=1)
        scaler       = RobustScaler()
        feats_scaled = np.full(feats_arr.shape, np.nan, dtype=np.float64)
        if valid_mask.sum() > 0:
            feats_scaled[valid_mask] = scaler.fit_transform(
                feats_arr[valid_mask].astype(np.float64)
            )

        other_valid   = other_mask & valid_mask
        n_other_valid = int(other_valid.sum())

        if n_other_valid >= n_gmm_components * 5:
            print(f"  Fitting GMM ({n_gmm_components} components on "
                  f"{n_other_valid:,} residual points) ...")
            gmm = fit_gmm(
                feats_scaled[other_mask],
                rule_labels[other_mask],
                n_components=n_gmm_components,
                random_state=random_state,
            )
            print(f"  GMM converged: {gmm.converged_}")
            g_c1, g_c2, g_p1 = gmm_predict(
                gmm,
                feats_scaled[other_mask],
                rule_labels[other_mask],
                CLASS_NAMES,
            )
            class_1[other_mask] = g_c1
            class_2[other_mask] = g_c2
            prob_1[other_mask]  = g_p1
        else:
            print(f"  ⚠  Too few residual points for GMM ({n_other_valid}) "
                  f"— labelling as 'other'")

    # ── class_2 for rule-assigned points (centroid distance) ──────────────
    _fill_class2_for_ruled(
        class_1, class_2, rule_labels, feats_arr, CLASS_NAMES, other_mask
    )

    # ── Assemble full result (all meta pids, including non-eligible) ───────
    cls_df = pd.DataFrame({
        'pid':           pids_arr,
        'class_1':       class_1,
        'class_2':       class_2,
        'class_prob_1':  prob_1,
        'periodic':      flags['periodic'],
        'jumpy':         flags['jumpy'],
        'variable_flag': flags['variable_flag'],
        'noisy_trend':   flags['noisy_trend'],
    })

    all_meta_pids = con.execute("SELECT pid FROM meta").df()["pid"]
    full_cls = all_meta_pids.to_frame().merge(cls_df, on='pid', how='left')
    full_cls['class_1']       = full_cls['class_1'].fillna('unclassified')
    full_cls['class_2']       = full_cls['class_2'].fillna('unclassified')
    full_cls['class_prob_1']  = full_cls['class_prob_1'].fillna(0.0).astype(np.float32)
    full_cls['periodic']      = full_cls['periodic'].fillna(False).astype(bool)
    full_cls['jumpy']         = full_cls['jumpy'].fillna(False).astype(bool)
    full_cls['variable_flag'] = full_cls['variable_flag'].fillna(False).astype(bool)
    full_cls['noisy_trend']   = full_cls['noisy_trend'].fillna(False).astype(bool)

    # ── Print distributions ────────────────────────────────────────────────
    print_class_distribution(full_cls['class_1'].values)
    flag_arrays = {
        'periodic':      full_cls['periodic'].values,
        'jumpy':         full_cls['jumpy'].values,
        'variable_flag': full_cls['variable_flag'].values,
        'noisy_trend':   full_cls['noisy_trend'].values,
    }
    print_flag_distribution(flag_arrays, n_total=len(full_cls))

    # ── Write back to meta parquet ─────────────────────────────────────────
    print(f"\n  Writing to {meta_path.name} ...")
    update_meta_parquet(
        meta_path,
        full_cls['pid'],
        full_cls['class_1'].values,
        full_cls['class_2'].values,
        full_cls['class_prob_1'].values,
        full_cls['periodic'].values,
        full_cls['jumpy'].values,
        full_cls['variable_flag'].values,
        full_cls['noisy_trend'].values,
    )

    elapsed = time.time() - t0
    print(f"  ✓ Done in {elapsed:.1f} s")
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EGMS time series classifier v3 — revised class scheme.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data-dir", required=True,
        help="Folder containing *_meta.parquet and *_ts.parquet pairs")

    # Filter
    parser.add_argument("--min-coh",  type=float, default=0.5,
        help="Minimum temporal coherence (default 0.5)")
    parser.add_argument("--min-vel",  type=float, default=0.0,
        help="Minimum |mean_velocity| mm/yr to include (default 0.0 = all)")

    # Primary classification thresholds
    parser.add_argument("--v-stable",     type=float, default=2.5,
        help="Max |velocity| for stable/noisy classes mm/yr (default 2.5)")
    parser.add_argument("--noise-thresh", type=float, default=3.0,
        help="RMSE detrended threshold (mm) separating stable from noisy (default 3.0)")
    parser.add_argument("--quad-r2",      type=float, default=0.05,
        help="Min quadratic R² improvement for accel/decel rule (default 0.05)")
    parser.add_argument("--accel-thresh", type=float, default=1.5,
        help="Min split-half velocity difference (mm/yr) for accel/decel (default 1.5)")

    # Descriptor flag thresholds
    parser.add_argument("--cusum-thresh", type=float, default=2.5,
        help="CUSUM score threshold for variable_flag descriptor (default 2.5)")
    parser.add_argument("--jump-mm",      type=float, default=5.0,
        help="Min step magnitude (mm) for jumpy flag (default 5.0; "
             "raise to 8-20 for Iceland/Scandinavia)")
    parser.add_argument("--seasonal-r2",  type=float, default=0.15,
        help="Min seasonal R² for periodic flag (default 0.15; "
             "lower than v2's 0.25 for better recall of moderate seasonality)")
    parser.add_argument("--noisy-trend",  type=float, default=4.0,
        help="RMSE deseason threshold (mm) for noisy_trend flag on moving points "
             "(default 4.0)")

    # GMM
    parser.add_argument("--n-components", type=int, default=5,
        help="GMM components for residual 'other' points (default 5 — "
             "matches the 5 primary classes; fewer than v2's 8 for speed)")
    parser.add_argument("--random-state", type=int, default=42,
        help="Random seed for GMM reproducibility (default 42)")

    # Performance
    parser.add_argument("--chunk-size", type=int, default=50_000,
        help="Points per chunk for feature extraction (default 50000)")
    parser.add_argument("--n-jobs", type=int, default=-1,
        help="Parallel workers for file processing (-1=all cores, 1=sequential, "
             "default -1).  Each worker processes one parquet pair independently.")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        print(f"ERROR: '{data_dir}' is not a valid directory")
        sys.exit(1)

    meta_files = sorted(data_dir.rglob("*_meta.parquet"))
    if not meta_files:
        print(f"ERROR: No *_meta.parquet files found under {data_dir}")
        sys.exit(1)

    print(f"\n{SEP}")
    print(f"  EGMS TIME SERIES CLASSIFIER  v3 (revised class scheme)")
    print(f"{SEP}")
    print(f"  Data dir        : {data_dir}")
    print(f"  Files           : {len(meta_files)}")
    print(f"\n  Primary class thresholds:")
    print(f"    stable max vel   : {args.v_stable} mm/yr")
    print(f"    noise RMSE       : {args.noise_thresh} mm")
    print(f"    accel threshold  : {args.accel_thresh} mm/yr  (split-half diff)")
    print(f"    quad R²          : {args.quad_r2}  (accel/decel improvement)")
    print(f"\n  Descriptor flag thresholds:")
    print(f"    jump min         : {args.jump_mm} mm  → jumpy flag")
    print(f"    CUSUM thresh     : {args.cusum_thresh}  → variable_flag")
    print(f"    seasonal R²      : {args.seasonal_r2}  → periodic flag")
    print(f"    noisy trend RMSE : {args.noisy_trend} mm  → noisy_trend flag")
    print(f"\n  GMM components  : {args.n_components}  (residual 'other' only)")
    print(f"  Chunk size      : {args.chunk_size:,}")
    print(f"  Filter          : coh≥{args.min_coh}, |vel|≥{args.min_vel} mm/yr")

    # ── Build work list (skip pairs with missing TS file) ───────────────────
    work_pairs = []
    for meta_path in meta_files:
        ts_path = meta_path.parent / meta_path.name.replace(
            "_meta.parquet", "_ts.parquet")
        if not ts_path.exists():
            print(f"\n  ✗ Missing TS file for {meta_path.name} — skipping")
            continue
        work_pairs.append((meta_path, ts_path))

    n_jobs   = args.n_jobs
    n_files  = len(work_pairs)
    n_cores  = os.cpu_count() or 1
    eff_jobs = n_cores if n_jobs == -1 else min(abs(n_jobs), n_cores)
    print(f"\n  Processing {n_files} file pairs "
          f"({eff_jobs} parallel workers) ...", flush=True)

    # Kwargs shared across all workers
    kw = dict(
        min_coh             = args.min_coh,
        min_vel             = args.min_vel,
        v_stable_max        = args.v_stable,
        noise_thresh        = args.noise_thresh,
        quad_r2_thresh      = args.quad_r2,
        accel_thresh        = args.accel_thresh,
        cusum_thresh        = args.cusum_thresh,
        jump_min_mm         = args.jump_mm,
        seasonal_r2_thresh  = args.seasonal_r2,
        noisy_trend_thresh  = args.noisy_trend,
        n_gmm_components    = args.n_components,
        chunk_size          = args.chunk_size,
        random_state        = args.random_state,
    )

    def _worker(pair):
        meta_path, ts_path = pair
        try:
            classify_pair(meta_path=meta_path, ts_path=ts_path, **kw)
            return (meta_path.name, None)
        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print(f"\n  ERROR processing {meta_path.name}: {e}\n{msg}",
                  flush=True)
            return (meta_path.name, str(e))

    if eff_jobs == 1 or n_files == 1:
        # Sequential — simpler output, easier to debug
        results = [_worker(p) for p in work_pairs]
    else:
        # Parallel across files using threading backend:
        # - Each worker has its own DuckDB in-memory connection (no sharing)
        # - threading avoids Windows process-spawn overhead
        # - GIL is released during numpy/pyarrow I/O so real parallelism occurs
        # - stdout from workers is interleaved but all output is preserved
        from joblib import Parallel, delayed as _delayed
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            _delayed(_worker)(p) for p in work_pairs
        )

    errors = [(name, err) for name, err in results if err is not None]

    print(f"\n{SEP}")
    print("  Classification complete.")
    print(f"  Files processed : {n_files - len(errors)}/{n_files}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for name, err in errors:
            print(f"    {name}: {err}")
    print(f"  New columns in *_meta.parquet:")
    print(f"    class_1, class_2, class_prob_1  (primary + secondary class)")
    print(f"    periodic, jumpy, variable_flag, noisy_trend  (descriptor flags)")
    print(f"  Bridge and viewer will auto-detect all new columns.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
