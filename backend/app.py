# app.py — Streamlit Isotope Dashboard (Multi-Spectrum, Peaks, Labels, Similarity + RF-on-Peaks)

from typing import List, Dict, Tuple, Optional
import io
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ================================
# Known isotope lines (keV)
# ================================
ISOTOPE_LINES = {
    "234U":  [53.2, 120.9],
    "235U":  [143.8, 185.7, 205.3],
    "238U":  [49.6, 113.5],
    "238Pu": [43.5, 99.9],
    "239Pu": [129.3, 203.5, 345.0, 375.0],
    "240Pu": [45.2, 104.2],
    "241Pu": [148.6, 208.0],
    "242Pu": [44.9, 103.7],
    "241Am": [59.5, 125.3, 332.4, 722.9],
}

# ================================
# Helpers
# ================================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def compute_energy_linear(channel: np.ndarray, kev_per_bin: float, one_based: bool) -> np.ndarray:
    ch0 = channel - 1 if one_based else channel
    return kev_per_bin * ch0

def rolling_weighted_centroid(energy: pd.Series, counts: pd.Series, window: int) -> pd.Series:
    eps = 1e-9
    wsum = counts.rolling(window, center=True, min_periods=1).sum()
    ewsum = (energy * counts).rolling(window, center=True, min_periods=1).sum()
    return ewsum / (wsum + eps)

def rolling_peak_height(counts: pd.Series, window: int) -> pd.Series:
    return counts.rolling(window, center=True, min_periods=1).median()

def rolling_peak_area(counts: pd.Series, window: int) -> pd.Series:
    return counts.rolling(window, center=True, min_periods=1).sum()

def approx_fwhm_keV(energy: np.ndarray, counts: np.ndarray, window: int) -> np.ndarray:
    """Simple, SciPy-free local FWHM estimate per bin."""
    n = len(counts)
    half = max(1, window // 2)
    fwhm = np.zeros(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = counts[lo:hi]
        if seg.size == 0:
            continue
        peak = float(np.max(seg))
        hm = 0.5 * peak

        # Left crossing
        left = i
        while left > lo and counts[left] >= hm:
            left -= 1
        if left < i:
            e_left = np.interp(hm, [counts[left], counts[left + 1]], [energy[left], energy[left + 1]])
        else:
            e_left = energy[i]

        # Right crossing
        right = i
        while right < hi - 1 and counts[right] >= hm:
            right += 1
        if right > i:
            e_right = np.interp(hm, [counts[right - 1], counts[right]], [energy[right - 1], energy[right]])
        else:
            e_right = energy[i]

        fwhm[i] = max(0.0, e_right - e_left)
    return fwhm

# ---------------- Label Matching ----------------
def match_isotopes_at_energy(energy_keV: float, tolerance_keV: float) -> Tuple[List[str], List[float]]:
    """Return (isotopes_matched, lines_matched) for a measured energy."""
    matches_iso, matches_line = [], []
    for iso, lines in ISOTOPE_LINES.items():
        for line in lines:
            if abs(energy_keV - line) <= tolerance_keV:
                matches_iso.append(iso)
                matches_line.append(line)
    return matches_iso, matches_line

def labels_for_series(energies: pd.Series, tolerance_keV: float) -> pd.Series:
    out = []
    for e in energies.to_numpy():
        iso_match, _ = match_isotopes_at_energy(float(e), tolerance_keV)
        out.append(", ".join(sorted(set(iso_match))) if iso_match else "—")
    return pd.Series(out, index=energies.index)

def prepare_features(
    df_raw: pd.DataFrame,
    kev_per_bin: float,
    one_based: bool,
    area_window: int,
    height_window: int,
    centroid_window: int,
    fwhm_window: int,
    tolerance_keV: float,
    duplicate_width: bool = False,
) -> pd.DataFrame:
    """Build deterministic features from Channel + Count/Counts, add Labels column."""
    df = normalize_headers(df_raw).copy()

    cols_lower = {c.lower(): c for c in df.columns}
    if "channel" not in cols_lower:
        raise ValueError("CSV must contain a 'Channel' column.")
    channel_col = cols_lower["channel"]

    count_col = None
    for cand in ("count", "counts"):
        if cand in cols_lower:
            count_col = cols_lower[cand]
            break
    if count_col is None:
        raise ValueError("CSV must contain a 'Count' or 'Counts' column.")

    df[channel_col] = coerce_numeric(df[channel_col])
    df[count_col] = coerce_numeric(df[count_col])
    df = df.dropna(subset=[channel_col, count_col]).sort_values(channel_col).reset_index(drop=True)

    df["Energy_keV"] = compute_energy_linear(df[channel_col].to_numpy(), kev_per_bin, one_based)

    counts_series = df[count_col]
    out = pd.DataFrame(
        {
            "Channel": df[channel_col].astype(int).values,
            "Counts": counts_series.values,
            "Energy_keV": df["Energy_keV"].values,
            "Peak_Area": rolling_peak_area(counts_series, area_window).values,
            "Peak_Height": rolling_peak_height(counts_series, height_window).values,
            "Centroid_keV": rolling_weighted_centroid(df["Energy_keV"], counts_series, centroid_window).values,
        }
    )
    out["FWHM_keV"] = approx_fwhm_keV(out["Energy_keV"].to_numpy(), out["Counts"].to_numpy(), fwhm_window)

    # Per-bin labels (for human reference — NOT used as model feature)
    out["Labels"] = labels_for_series(out["Energy_keV"], tolerance_keV)

    if duplicate_width:
        out["FWHM_keV_dup"] = out["FWHM_keV"].values

    return out

# ---------------- Peak Finding ----------------
def find_peaks_simple(counts: np.ndarray, min_prominence: float, min_distance: int) -> np.ndarray:
    """
    Local-max peak picker:
      - peak > neighbors and ≥ (median + min_prominence * std)
      - keep peaks spaced by at least min_distance bins (greedy by height)
    """
    n = len(counts)
    if n < 3:
        return np.array([], dtype=int)

    med = float(np.median(counts))
    std = float(np.std(counts, ddof=0))
    thresh = med + min_prominence * std

    # initial candidates
    candidates = []
    for i in range(1, n - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1] and counts[i] >= thresh:
            candidates.append(i)
    if not candidates:
        return np.array([], dtype=int)

    # spacing by height
    candidates = sorted(candidates, key=lambda i: counts[i], reverse=True)
    selected, taken = [], np.zeros(n, dtype=bool)
    for idx in candidates:
        if taken[max(0, idx - min_distance):min(n, idx + min_distance + 1)].any():
            continue
        selected.append(idx)
        taken[max(0, idx - min_distance):min(n, idx + min_distance + 1)] = True

    selected.sort()
    return np.array(selected, dtype=int)

# ---------------- Similarity ----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.linalg.norm(a)
    bb = np.linalg.norm(b)
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (aa * bb))

def vectorize_counts_by_channel(df_feat: pd.DataFrame, channel_min: Optional[int] = None, channel_max: Optional[int] = None):
    """Return dense vector over a specified channel range (default: min..max of df)."""
    ch = df_feat["Channel"].astype(int).to_numpy()
    ct = df_feat["Counts"].to_numpy()

    if channel_min is None: channel_min = int(ch.min())
    if channel_max is None: channel_max = int(ch.max())
    length = channel_max - channel_min + 1
    vec = np.zeros(length, dtype=float)

    # Shift channels to [0..length-1]
    valid = (ch >= channel_min) & (ch <= channel_max)
    chv, ctv = ch[valid], ct[valid]
    idx = (chv - channel_min).astype(int)
    vec[idx] = ctv
    return vec, channel_min, channel_max

def similarity_to_others(active_name: str, feats_map: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Compute cosine similarity between active spectrum and others over the union channel range."""
    results = []
    if active_name not in feats_map:
        return results

    base = feats_map[active_name]
    for name, df in feats_map.items():
        if name == active_name:
            continue
        ch_min = int(min(base["Channel"].min(), df["Channel"].min()))
        ch_max = int(max(base["Channel"].max(), df["Channel"].max()))
        a, _, _ = vectorize_counts_by_channel(base, ch_min, ch_max)
        b, _, _ = vectorize_counts_by_channel(df, ch_min, ch_max)
        sim = cosine_similarity(a, b)
        results.append({"Reference": name, "Cosine_Similarity": sim})
    results.sort(key=lambda x: x["Cosine_Similarity"], reverse=True)
    return results

# ---------------- Model loading / alignment ----------------
def try_load_pickle(file_bytes: bytes):
    try:
        return pickle.loads(file_bytes)
    except Exception:
        import joblib
        bio = io.BytesIO(file_bytes)
        return joblib.load(bio)

def align_and_scale_for_model(
    df_feat: pd.DataFrame,
    feature_columns: Optional[List[str]],
    scaler_obj
):
    # default training order if JSON not provided
    default_cols = ["Counts", "Channel", "Energy_keV", "Peak_Area", "Peak_Height", "Centroid_keV", "FWHM_keV"]
    cols = feature_columns if feature_columns else default_cols
    X = df_feat.reindex(columns=cols)
    if X.isnull().any().any():
        raise ValueError("Some required model features are missing. Needed: " + ", ".join(cols))
    Xv = X.values
    if scaler_obj is not None:
        Xv = scaler_obj.transform(Xv)
    return Xv

# ================================
# UI
# ================================
st.set_page_config(page_title="Isotope Dashboard (Multi)", layout="wide")
st.title("Isotope Dashboard — Multi-Spectrum • Peaks • Labels • Similarity • RF-on-Peaks")

with st.sidebar:
    st.header("Calibration")
    one_based = st.checkbox("Channels start at 1", value=True)
    default_kev_per_bin = 3000.0 / 8191.0
    kev_per_bin = st.number_input("keV per bin", value=float(default_kev_per_bin), step=0.0001, format="%.6f")

    st.header("Windows")
    area_w = st.slider("Area window (bins)", 5, 101, 7, step=2)
    height_w = st.slider("Height window (bins)", 3, 101, 5, step=2)
    centroid_w = st.slider("Centroid window (bins)", 5, 101, 7, step=2)
    fwhm_w = st.slider("FWHM window (bins)", 5, 201, 31, step=2)

    st.header("Peak Finder")
    min_prom = st.slider("Min prominence (σ)", 0.0, 8.0, 2.0, 0.1)
    min_dist = st.slider("Min distance between peaks (bins)", 1, 200, 10, step=1)

    st.header("Label Match")
    tol_keV = st.number_input("Energy tolerance ± (keV)", value=1.0, step=0.1)

    st.header("Metadata")
    acq_date = st.date_input("Acquisition Date (optional)")

    st.header("Optional: Model Artifacts (RF on peaks)")
    model_file = st.file_uploader("RandomForest model (.pkl/.joblib/.keras)", type=["pkl","joblib","pickle","keras"])
    scaler_file = st.file_uploader("Scaler (scaler.pkl)", type=["pkl","joblib","pickle"])
    label_file  = st.file_uploader("LabelEncoder (label_encoder.pkl)", type=["pkl","joblib","pickle"])
    feat_json   = st.file_uploader("feature_columns.json (optional)", type=["json"])

# 1) Upload CSVs
st.markdown("### 1) Upload spectrum CSV files (one or more)")
files = st.file_uploader("Upload CSV(s) with Channel and Count/Counts", type=["csv"], accept_multiple_files=True)

if not files:
    st.info("Upload at least one spectrum CSV to begin.")
    st.stop()

# Parse all
raw_map: Dict[str, pd.DataFrame] = {}
for f in files:
    try:
        df = pd.read_csv(f)
        raw_map[f.name] = df
    except Exception as e:
        st.error(f"{f.name}: could not read CSV ({e})")

if not raw_map:
    st.stop()

names = list(raw_map.keys())
active_name = st.selectbox("Active spectrum to analyze", names, index=0)

st.markdown("#### Raw table (active)")
st.dataframe(raw_map[active_name].head(12), use_container_width=True)

# 2) Compute features for all
feat_map: Dict[str, pd.DataFrame] = {}
errors = []
for name, df in raw_map.items():
    try:
        feat_map[name] = prepare_features(
            df,
            kev_per_bin=kev_per_bin,
            one_based=one_based,
            area_window=area_w,
            height_window=height_w,
            centroid_window=centroid_w,
            fwhm_window=fwhm_w,
            tolerance_keV=tol_keV,
            duplicate_width=False,
        )
    except Exception as e:
        errors.append(f"{name}: {e}")
if errors:
    st.error("Some files failed feature computation:\n" + "\n".join(errors))
if active_name not in feat_map:
    st.stop()

features = feat_map[active_name]

# 3) Metadata
with st.container():
    st.markdown("### 2) Metadata")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Active Spectrum", active_name)
    meta_cols[1].metric("Rows", f"{len(features):,}")
    meta_cols[2].metric("Acquisition Date", str(acq_date) if acq_date else "—")

# 4) Features preview
st.markdown("### 3) Computed features (first 20 rows)")
st.dataframe(features.head(20), use_container_width=True)

# 5) Graph
with st.expander("Counts vs Channel (active spectrum)"):
    import altair as alt
    base_chart = (
        alt.Chart(features[["Channel", "Counts"]])
        .mark_line()
        .encode(
            x=alt.X("Channel:Q"),
            y=alt.Y("Counts:Q"),
            tooltip=["Channel", "Counts"],
        )
        .properties(height=320, width="container")
    )
    st.altair_chart(base_chart, use_container_width=True)

# 6) Peak detection + isotope labeling
st.markdown("---")
st.markdown("### 4) Peak Detection & Isotope Labeling (active)")

counts = features["Counts"].to_numpy()
energies = features["Energy_keV"].to_numpy()
peak_idx = find_peaks_simple(counts, min_prominence=min_prom, min_distance=min_dist)

peak_rows = []
if peak_idx.size == 0:
    st.warning("No peaks found with the current settings. Try lowering prominence or min distance.")
else:
    for i in peak_idx:
        e = float(energies[i])
        iso_match, line_match = match_isotopes_at_energy(e, tol_keV)
        peak_rows.append(
            {
                "Channel": int(features.loc[i, "Channel"]),
                "Energy_keV": e,
                "Counts": float(counts[i]),
                "FWHM_keV": float(features.loc[i, "FWHM_keV"]),
                "Matched_Isotopes": ", ".join(sorted(set(iso_match))) if iso_match else "—",
                "Matched_Lines_keV": ", ".join(f"{lm:.1f}" for lm in sorted(set(line_match))) if line_match else "—",
            }
        )
    peak_df = pd.DataFrame(peak_rows).sort_values("Counts", ascending=False).reset_index(drop=True)

    # overlay peaks on graph
    with st.expander("Overlay: Detected peaks on Counts vs Channel"):
        import altair as alt
        peaks_overlay = (
            alt.Chart(pd.DataFrame({"Channel": features["Channel"].iloc[peak_idx],
                                    "Counts": features["Counts"].iloc[peak_idx]}))
            .mark_point(size=60)
            .encode(x="Channel:Q", y="Counts:Q", tooltip=["Channel", "Counts"])
        )
        st.altair_chart(base_chart + peaks_overlay, use_container_width=True)

    st.markdown("#### Peak Report")
    st.dataframe(peak_df, use_container_width=True)

    # 6a) OPTIONAL: Run RF model on peaks only
    if model_file is not None:
        try:
            model = try_load_pickle(model_file.read())
            st.success(f"Loaded model: {type(model)}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model = None

        scaler_obj = None
        if scaler_file is not None:
            try:
                scaler_obj = try_load_pickle(scaler_file.read())
                st.info("Scaler loaded.")
            except Exception as e:
                st.error(f"Failed to load scaler: {e}")

        label_enc = None
        if label_file is not None:
            try:
                label_enc = try_load_pickle(label_file.read())
                st.info("LabelEncoder loaded.")
            except Exception as e:
                st.error(f"Failed to load label encoder: {e}")

        feature_columns = None
        if feat_json is not None:
            try:
                feature_columns = json.load(feat_json)
                st.info(f"feature_columns.json loaded: {feature_columns}")
            except Exception as e:
                st.error(f"Failed to parse feature_columns.json: {e}")

        if model is not None:
            try:
                # Build the feature matrix for just the peaks (exclude non-numeric 'Labels')
                numeric_feat = features.drop(columns=[c for c in ["Labels"] if c in features.columns])
                X_peaks_all = numeric_feat.iloc[peak_idx]  # rows at peak indices
                Xv = align_and_scale_for_model(X_peaks_all, feature_columns, scaler_obj)

                y_pred = model.predict(Xv)
                # probabilities if available
                proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(Xv)
                    except Exception:
                        proba = None

                # inverse transform labels if encoder provided
                if label_enc is not None and hasattr(label_enc, "inverse_transform"):
                    try:
                        y_disp = label_enc.inverse_transform(y_pred)
                    except Exception:
                        y_disp = y_pred
                else:
                    y_disp = y_pred

                peak_df["RF_Predicted_Class"] = list(map(str, y_disp))
                if proba is not None:
                    # highest probability
                    top_p = np.max(proba, axis=1)
                    peak_df["RF_Confidence"] = top_p
                st.markdown("#### Random Forest Classification (on detected peaks)")
                st.dataframe(peak_df, use_container_width=True)
            except Exception as e:
                st.warning(f"RF prediction on peaks failed: {e}")

    # Isotope summary
    summary: Dict[str, List[float]] = {}
    for _, row in peak_df.iterrows():
        if row["Matched_Isotopes"] == "—":
            continue
        isotopes = [s.strip() for s in row["Matched_Isotopes"].split(",")]
        if row["Matched_Lines_keV"] != "—":
            lines = [float(x) for x in row["Matched_Lines_keV"].split(",")]
        else:
            lines = []
        for iso in isotopes:
            summary.setdefault(iso, [])
            summary[iso].extend(lines)

    if summary:
        summ_rows = []
        for iso, lines in summary.items():
            uniq = sorted(set(round(x, 1) for x in lines))
            summ_rows.append(
                {"Isotope": iso, "Matched_Lines_keV": ", ".join(f"{x:.1f}" for x in uniq), "Count_of_Lines": len(uniq)}
            )
        st.markdown("#### Isotope Summary (present based on matched peaks)")
        st.dataframe(pd.DataFrame(summ_rows).sort_values(["Count_of_Lines", "Isotope"], ascending=[False, True]),
                     use_container_width=True)
    else:
        st.info("No isotope lines matched the detected peaks under the current tolerance.")

# 7) Similarity to other uploaded spectra
st.markdown("---")
st.markdown("### 5) Similarity — Active vs Other Uploaded Spectra")
if len(feat_map) == 1:
    st.info("Upload more than one file to see similarity comparisons.")
else:
    sims = similarity_to_others(active_name, feat_map)
    if sims:
        st.dataframe(pd.DataFrame(sims), use_container_width=True)
    else:
        st.info("Could not compute similarity (empty/degenerate vectors).")

# 8) Downloads
st.markdown("---")
st.markdown("### 6) Downloads")
if len(peak_rows) > 0:
    out_peaks = pd.DataFrame(peak_rows).copy()
    out_peaks["Acquisition_Date"] = str(acq_date) if acq_date else ""
    st.download_button(
        "⬇️ Download Peak Report (CSV)",
        data=out_peaks.to_csv(index=False).encode("utf-8"),
        file_name=f"peak_report_{active_name}.csv",
        mime="text/csv",
    )
st.download_button(
    "⬇️ Download Features (active, CSV)",
    data=features.to_csv(index=False).encode("utf-8"),
    file_name=f"features_{active_name}.csv",
    mime="text/csv",
)
