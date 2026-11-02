"""
Credit Risk Prediction App (Streamlit)
=======================================
Dataset  : data_clean.csv (aplikasi kartu kredit, target 1 = berisiko / bad client)
Fitur    : 1) Single prediction (input widget)
           2) Batch prediction (upload CSV)
           3) Interpretasi model (global & lokal, model-agnostic)

Jalankan:
    pip install "streamlit>=1.50" pandas numpy plotly scikit-learn joblib
    streamlit run app.py

Catatan:
- Bagian MODEL masih placeholder (lihat blok "PLACEHOLDER MODEL" di bawah).
  Selama model belum dipasang, app berjalan dalam MODE DEMO memakai skor dummy
  supaya semua fitur UI & interpretasi tetap bisa dicoba.
- Interpretasi dibuat model-agnostic (hanya butuh fungsi predict_proba),
  jadi tetap berfungsi setelah model diganti apa pun (LogReg, RF, XGBoost, dll).
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score
import pickle
import requests
import json

# =============================================================================
# KONFIGURASI
# ========================= ====================================================
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")

DATA_PATH = "data/cleaned_data.csv"      # data referensi (opsi widget, background interpretasi)
MODEL_PATH = "model_rf_v2.sav"          # lokasi model final nanti
TARGET_COL = "TARGET"
ID_COL = "ID"
RANDOM_STATE = 42

NUM_COLS = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
]
CAT_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
FEATURE_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE",
    "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
]

# Label yang lebih mudah dibaca untuk tampilan
FEATURE_LABELS = {
    "CODE_GENDER": "Jenis kelamin",
    "FLAG_OWN_CAR": "Punya mobil",
    "FLAG_OWN_REALTY": "Punya properti",
    "CNT_CHILDREN": "Jumlah anak",
    "AMT_INCOME_TOTAL": "Pendapatan tahunan",
    "NAME_INCOME_TYPE": "Jenis pendapatan",
    "NAME_EDUCATION_TYPE": "Pendidikan",
    "NAME_FAMILY_STATUS": "Status keluarga",
    "NAME_HOUSING_TYPE": "Jenis tempat tinggal",
    "DAYS_BIRTH": "Usia (hari)",
    "DAYS_EMPLOYED": "Lama bekerja (hari)",
    "FLAG_MOBIL": "Punya HP",
    "FLAG_WORK_PHONE": "Punya telepon kantor",
    "FLAG_PHONE": "Punya telepon rumah",
    "FLAG_EMAIL": "Punya email",
    "OCCUPATION_TYPE": "Pekerjaan",
    "CNT_FAM_MEMBERS": "Jumlah anggota keluarga",
}


def label(col):
    return FEATURE_LABELS.get(col, col)


# =============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>  PLACEHOLDER MODEL  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Ganti 3 fungsi di bawah ini saat model sudah siap. Bagian lain tidak perlu diubah.
# =============================================================================
@st.cache_resource
def load_model():
    """
    TODO: muat model final di sini, lalu return objek modelnya.
    Contoh:
        import joblib
        return joblib.load(MODEL_PATH)

    Selama return None -> app berjalan dalam MODE DEMO (skor dummy).
    """
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


def preprocess(df):
    """
    TODO: feature engineering / preprocessing yang TIDAK ada di dalam pipeline model.
    Input  : DataFrame dengan kolom FEATURE_COLS (format mentah seperti data_clean.csv)
    Output : DataFrame/array yang siap masuk ke model.predict_proba()
    Kalau model Anda sudah berupa sklearn Pipeline lengkap, biarkan return df apa adanya.
    """
    return df


def model_predict_proba(model, df):
    """
    Mengembalikan array 1D probabilitas kelas 1 (berisiko) untuk setiap baris df.
    TODO: sesuaikan jika model Anda punya API berbeda (mis. model.predict() untuk regresi).
    """
    X = preprocess(df[FEATURE_COLS].copy())
    return np.asarray(model.predict_proba(X))[:, 1]

def get_prediction_from_api(df):
    payload = json.loads(df.to_json(orient='records'))
    res = requests.post('https://ml-deployment-umber.vercel.app/predict', json=payload)
    prediction = res.json()
    return np.array(prediction['prob'])
# -----------------------------------------------------------------------------
# Skor dummy (hanya untuk MODE DEMO). Boleh dihapus setelah model asli dipasang.
# -----------------------------------------------------------------------------
def dummy_predict_proba(df):
    age_years = df["DAYS_BIRTH"].astype(float) / 365.25
    emp_years = df["DAYS_EMPLOYED"].astype(float) / 365.25
    income = np.log1p(df["AMT_INCOME_TOTAL"].astype(float))
    z = (
        -3.6
        - 0.35 * (income - 12.0)
        - 0.06 * np.clip(emp_years, 0, 20)
        - 0.015 * (age_years - 40)
        + 0.20 * df["CNT_CHILDREN"].astype(float)
        + 0.30 * (df["FLAG_OWN_REALTY"] == "N")
        + 0.15 * (df["FLAG_OWN_CAR"] == "N")
        + 0.40 * df["NAME_FAMILY_STATUS"].isin(["Separated", "Widow"])
        + 0.30 * df["NAME_HOUSING_TYPE"].isin(["Rented apartment", "With parents"])
        + 0.25 * (df["NAME_EDUCATION_TYPE"] == "Lower secondary")
        + 0.25 * df["OCCUPATION_TYPE"].isin(["Low-skill Laborers", "Drivers", "Laborers"])
    )
    return (1 / (1 + np.exp(-z))).to_numpy()


# =============================================================================
# FUNGSI PREDIKSI UTAMA (dipakai di seluruh app)
# =============================================================================
def predict_proba(df, use_api:bool=True):
    model = load_model()
    if model is None:
        return dummy_predict_proba(df)
    if use_api is True:
        return get_prediction_from_api(df)
    else:
        return model_predict_proba(model, df)


def is_demo_mode():
    return load_model() is None


# =============================================================================
# DATA
# =============================================================================
@st.cache_data
def load_reference_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df


def get_reference_data():
    if os.path.exists(DATA_PATH):
        return load_reference_data(DATA_PATH)
    st.sidebar.warning(f"`{DATA_PATH}` tidak ditemukan. Upload data referensi:")
    up = st.sidebar.file_uploader("Data referensi (data_clean.csv)", type="csv", key="ref_upload")
    if up is None:
        st.info("Upload data referensi di sidebar untuk memulai.")
        st.stop()
    return load_reference_data(up)


def validate_columns(df):
    return [c for c in FEATURE_COLS if c not in df.columns]


def coerce_types(df):
    """Samakan tipe data input dengan data training."""
    df = df.copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df


def risk_label(p, threshold):
    return np.where(p >= threshold, "Berisiko (1)", "Tidak berisiko (0)")


# =============================================================================
# INTERPRETABILITY (model-agnostic)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_background(ref_df, n=100):
    """Sampel data referensi sebagai 'baseline' untuk explanation lokal."""
    return ref_df[FEATURE_COLS].sample(min(n, len(ref_df)), random_state=RANDOM_STATE).reset_index(drop=True)


def local_contributions(rows, background):
    """
    Kontribusi tiap fitur terhadap prediksi (metode occlusion / marginal replacement):
        kontribusi_j = p(x) - rata-rata_b[ p(x dengan fitur j diganti nilai dari baris background b) ]
    Positif  -> fitur tsb MENAIKKAN risiko dibanding kondisi rata-rata
    Negatif  -> fitur tsb MENURUNKAN risiko
    rows: DataFrame (1 atau beberapa baris). Return DataFrame (n_rows x n_features).
    """
    rows = rows[FEATURE_COLS].reset_index(drop=True)
    n_rows, n_bg = len(rows), len(background)
    base_pred = predict_proba(rows)
    # Replikasi: setiap baris diulang sebanyak n_bg
    rep = rows.loc[rows.index.repeat(n_bg)].reset_index(drop=True)
    bg_tiled = pd.concat([background] * n_rows, ignore_index=True)
    contrib = {}
    for col in FEATURE_COLS:
        tmp = rep.copy()
        tmp[col] = bg_tiled[col].to_numpy()
        p = predict_proba(tmp).reshape(n_rows, n_bg).mean(axis=1)
        contrib[col] = base_pred - p
    return pd.DataFrame(contrib)


@st.cache_data(show_spinner=False)
def permutation_importance(ref_df, n_repeats=3, n_neg=4000, _model_key=None):
    """
    Global importance: seberapa besar ROC-AUC turun jika satu fitur diacak.
    Memakai semua baris positif + sampel baris negatif agar cepat & stabil.
    """
    if TARGET_COL not in ref_df.columns:
        return None, None
    pos = ref_df[ref_df[TARGET_COL] == 1]
    neg = ref_df[ref_df[TARGET_COL] == 0]
    sample = pd.concat([pos, neg.sample(min(n_neg, len(neg)), random_state=RANDOM_STATE)])
    sample = sample.reset_index(drop=True)
    X, y = sample[FEATURE_COLS], sample[TARGET_COL].to_numpy()
    base_auc = roc_auc_score(y, predict_proba(X))
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for col in FEATURE_COLS:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base_auc - roc_auc_score(y, predict_proba(Xp)))
        rows.append({"feature": col, "importance": np.mean(drops), "std": np.std(drops)})
    imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
    imp["label"] = imp["feature"].map(label)
    return imp, base_auc


@st.cache_data(show_spinner=False)
def partial_dependence(ref_df, feature, n_sample=500, n_grid=20, _model_key=None):
    """Rata-rata prediksi ketika satu fitur di-set ke nilai tertentu (PDP)."""
    X = ref_df[FEATURE_COLS].sample(min(n_sample, len(ref_df)), random_state=RANDOM_STATE)
    if feature in CAT_COLS:
        grid = ref_df[feature].dropna().value_counts().index.tolist()
    else:
        vals = ref_df[feature].dropna()
        grid = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, n_grid)))
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        out.append({"value": g, "avg_proba": predict_proba(Xg).mean()})
    return pd.DataFrame(out)


def plot_contribution_bar(contrib_series, top_n=10, title="Kontribusi fitur"):
    s = contrib_series.reindex(contrib_series.abs().sort_values(ascending=False).index)[:top_n]
    s = s.iloc[::-1]  # terbesar di atas
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=[label(i) for i in s.index], orientation="h", marker_color=colors,
        text=[f"{v:+.2%}" for v in s.values], textposition="outside", cliponaxis=False,
    ))
    lo, hi = min(s.min(), 0), max(s.max(), 0)
    pad = (hi - lo) * 0.25 or 0.01
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    fig.update_layout(
        title=title, xaxis_title="Perubahan probabilitas risiko (poin %)",
        xaxis_tickformat=".1%", height=60 + 35 * len(s), margin=dict(l=10, r=40, t=50, b=40),
    )
    return fig


def plot_gauge(p, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".2f"},
        title={"text": "Probabilitas berisiko"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d62728" if p >= threshold else "#2ca02c"},
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =============================================================================
# UI: SIDEBAR
# =============================================================================
ref = get_reference_data()
missing_ref = validate_columns(ref)
if missing_ref:
    st.error(f"Data referensi tidak memiliki kolom: {missing_ref}")
    st.stop()
ref = coerce_types(ref)
background = get_background(ref)
model_key = "demo" if is_demo_mode() else MODEL_PATH  # untuk invalidasi cache

st.sidebar.title("⚙️ Pengaturan")
threshold = st.sidebar.slider(
    "Threshold klasifikasi", 0.01, 0.99, 0.10, 0.01,
    help="Probabilitas ≥ threshold → diklasifikasikan Berisiko (1). "
         "Data sangat imbalanced (~1.7% kelas 1), jadi threshold sering perlu lebih rendah dari 0.5.",
)
top_n = st.sidebar.slider("Jumlah fitur ditampilkan", 5, len(FEATURE_COLS), 10)
st.sidebar.markdown("---")
if is_demo_mode():
    st.sidebar.warning("🧪 **MODE DEMO** — model belum dipasang. Prediksi memakai skor dummy.")
else:
    st.sidebar.success("✅ Model aktif")

# =============================================================================
# UI: HEADER
# =============================================================================
st.title("💳 Credit Risk Prediction")
st.caption("Prediksi apakah pemohon kartu kredit berisiko (target = 1) beserta faktor yang paling berpengaruh.")
if is_demo_mode():
    st.warning("Model belum dipasang (placeholder). Ganti fungsi `load_model()` di script untuk memakai model asli.")

tab_single, tab_batch, tab_global = st.tabs(
    ["🧍 Single Prediction", "📂 Batch Prediction", "🔍 Interpretasi Model"]
)


# =============================================================================
# TAB 1: SINGLE PREDICTION
# =============================================================================
def options(col):
    return sorted(ref[col].dropna().unique().tolist())


with tab_single:
    st.subheader("Input data pemohon")
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profil**")
            gender = st.selectbox("Jenis kelamin", options("CODE_GENDER"),
                                  format_func=lambda x: {"M": "Laki-laki", "F": "Perempuan"}.get(x, x))
            age = st.number_input("Usia (tahun)", 18, 80, 40)
            education = st.selectbox(label("NAME_EDUCATION_TYPE"), options("NAME_EDUCATION_TYPE"))
            family = st.selectbox(label("NAME_FAMILY_STATUS"), options("NAME_FAMILY_STATUS"))
            children = st.number_input(label("CNT_CHILDREN"), 0, 20, 0)
            fam_members = st.number_input(label("CNT_FAM_MEMBERS"), 1, 20, 2)
        with c2:
            st.markdown("**Pekerjaan & Pendapatan**")
            income = st.number_input(label("AMT_INCOME_TOTAL"), min_value=0.0,
                                     value=float(ref["AMT_INCOME_TOTAL"].median()), step=5000.0)
            income_type = st.selectbox(label("NAME_INCOME_TYPE"), options("NAME_INCOME_TYPE"))
            occ_opts = ["(Tidak diketahui)"] + options("OCCUPATION_TYPE")
            occupation = st.selectbox(label("OCCUPATION_TYPE"), occ_opts)
            emp_years = st.number_input("Lama bekerja (tahun)", 0.0, 60.0, 5.0, 0.5,
                                        help="Isi 0 jika tidak bekerja / pensiun.")
        with c3:
            st.markdown("**Aset & Kontak**")
            own_car = st.radio("Punya mobil?", ["Y", "N"], horizontal=True)
            own_realty = st.radio("Punya properti?", ["Y", "N"], horizontal=True)
            housing = st.selectbox(label("NAME_HOUSING_TYPE"), options("NAME_HOUSING_TYPE"))
            work_phone = st.checkbox("Punya telepon kantor")
            phone = st.checkbox("Punya telepon rumah")
            email = st.checkbox("Punya email")
        submitted = st.form_submit_button("🔮 Prediksi", type="primary", width="stretch")

    if submitted:
        input_df = pd.DataFrame([{
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family,
            "NAME_HOUSING_TYPE": housing,
            "DAYS_BIRTH": int(round(age * 365.25)),
            "DAYS_EMPLOYED": int(round(emp_years * 365.25)),
            "FLAG_MOBIL": 1,
            "FLAG_WORK_PHONE": int(work_phone),
            "FLAG_PHONE": int(phone),
            "FLAG_EMAIL": int(email),
            "OCCUPATION_TYPE": np.nan if occupation == "(Tidak diketahui)" else occupation,
            "CNT_FAM_MEMBERS": float(fam_members),
        }])
        input_df = coerce_types(input_df)
        st.session_state["single_input"] = input_df

    if "single_input" in st.session_state:
        input_df = st.session_state["single_input"]
        proba = float(predict_proba(input_df)[0])
        pred = risk_label(np.array([proba]), threshold)[0]
        avg_proba = float(predict_proba(background).mean())

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.plotly_chart(plot_gauge(proba, threshold), width="stretch")
            if proba >= threshold:
                st.error(f"**Hasil: {pred}**")
            else:
                st.success(f"**Hasil: {pred}**")
            st.metric("Dibanding rata-rata populasi", f"{proba:.2%}", f"{proba - avg_proba:+.2%}",
                      delta_color="inverse")
        with r2:
            with st.spinner("Menghitung kontribusi fitur..."):
                contrib = local_contributions(input_df, background).iloc[0]
            st.plotly_chart(
                plot_contribution_bar(contrib, top_n, "Faktor yang paling memengaruhi prediksi ini"),
                width="stretch",
            )
            st.caption("🔴 Merah = menaikkan risiko  ·  🟢 Hijau = menurunkan risiko "
                       "(dibanding nilai tipikal di data referensi).")

        up = contrib[contrib > 0].sort_values(ascending=False).head(3)
        down = contrib[contrib < 0].sort_values().head(3)
        e1, e2 = st.columns(2)
        e1.markdown("**Faktor penaik risiko utama:**\n" +
                    ("\n".join(f"- {label(k)} ({v:+.2%})" for k, v in up.items()) or "- (tidak ada)"))
        e2.markdown("**Faktor penurun risiko utama:**\n" +
                    ("\n".join(f"- {label(k)} ({v:+.2%})" for k, v in down.items()) or "- (tidak ada)"))
        with st.expander("Lihat data input (format model)"):
            st.dataframe(input_df, width="stretch")


# =============================================================================
# TAB 2: BATCH PREDICTION
# =============================================================================
with tab_batch:
    st.subheader("Upload CSV untuk prediksi massal")
    st.markdown(
        f"File harus memiliki kolom: `{', '.join(FEATURE_COLS)}`. "
        f"Kolom `{ID_COL}` dan `{TARGET_COL}` opsional (jika `{TARGET_COL}` ada, evaluasi ditampilkan)."
    )
    template = ref[[ID_COL] + FEATURE_COLS].head(5) if ID_COL in ref.columns else ref[FEATURE_COLS].head(5)
    st.download_button("⬇️ Download template CSV", template.to_csv(index=False).encode(),
                       "template_batch.csv", "text/csv")

    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch_upload")
    batch = None
    if batch_file is not None:
        batch = pd.read_csv(batch_file)
        missing = validate_columns(batch)
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {missing}")
            batch = None  # jangan st.stop() agar tab lain tetap tampil
    if batch is not None:
        batch = coerce_types(batch)
        n_bad_rows = batch[NUM_COLS].isna().any(axis=1).sum()
        if n_bad_rows:
            st.warning(f"{n_bad_rows} baris punya nilai numerik kosong/tidak valid. "
                       "Pastikan model/preprocess Anda bisa menanganinya.")

        with st.spinner("Memprediksi..."):
            proba = predict_proba(batch)
        result = batch.copy()
        result["proba_berisiko"] = proba
        result["prediksi"] = (proba >= threshold).astype(int)
        result["label_prediksi"] = risk_label(proba, threshold)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah baris", f"{len(result):,}")
        m2.metric("Diprediksi berisiko", f"{result['prediksi'].sum():,}")
        m3.metric("% berisiko", f"{result['prediksi'].mean():.2%}")
        m4.metric("Rata-rata probabilitas", f"{proba.mean():.2%}")

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(result, x="proba_berisiko", nbins=50,
                               title="Distribusi probabilitas berisiko")
            fig.add_vline(x=threshold, line_dash="dash", annotation_text="threshold")
            st.plotly_chart(fig, width="stretch")
        with g2:
            seg_col = st.selectbox("Rata-rata risiko per segmen", CAT_COLS, format_func=label)
            seg = (result.assign(**{seg_col: result[seg_col].fillna("(kosong)")})
                   .groupby(seg_col)["proba_berisiko"].mean().sort_values().reset_index())
            fig = px.bar(seg, x="proba_berisiko", y=seg_col, orientation="h",
                         title=f"Rata-rata probabilitas per {label(seg_col)}")
            fig.update_layout(xaxis_tickformat=".1%", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")

        # Evaluasi jika target tersedia
        if TARGET_COL in batch.columns and batch[TARGET_COL].nunique() == 2:
            st.markdown("#### 📏 Evaluasi (kolom target ditemukan)")
            y = batch[TARGET_COL].astype(int)
            tn, fp, fn, tp = confusion_matrix(y, result["prediksi"], labels=[0, 1]).ravel()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}")
            k2.metric("Precision", f"{tp / (tp + fp):.3f}" if (tp + fp) else "-")
            k3.metric("Recall", f"{tp / (tp + fn):.3f}" if (tp + fn) else "-")
            k4.metric("Accuracy", f"{(tp + tn) / len(y):.3f}")
            cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Aktual 0", "Aktual 1"],
                              columns=["Prediksi 0", "Prediksi 1"])
            st.dataframe(cm, width="content")

        st.markdown("#### Hasil prediksi")
        st.dataframe(result.sort_values("proba_berisiko", ascending=False),
                     width="stretch", height=350)
        st.download_button("⬇️ Download hasil (CSV)", result.to_csv(index=False).encode(),
                           "hasil_prediksi.csv", "text/csv", type="primary")

        # Interpretasi per baris & ringkasan batch
        st.markdown("#### 🔍 Interpretasi hasil batch")
        b1, b2 = st.columns(2)
        with b1:
            idx_options = result.sort_values("proba_berisiko", ascending=False).index.tolist()
            show_id = ID_COL in result.columns
            row_idx = st.selectbox(
                "Pilih baris untuk dijelaskan (urut dari risiko tertinggi)", idx_options,
                format_func=lambda i: (f"{ID_COL} {result.loc[i, ID_COL]}" if show_id else f"Baris {i}")
                + f" — {result.loc[i, 'proba_berisiko']:.2%}",
            )
            contrib_row = local_contributions(batch.loc[[row_idx]], background).iloc[0]
            st.plotly_chart(plot_contribution_bar(contrib_row, top_n, "Kontribusi fitur (baris terpilih)"),
                            width="stretch")
        with b2:
            max_rows = 300
            sample_b = batch.sample(min(max_rows, len(batch)), random_state=RANDOM_STATE)
            with st.spinner("Menghitung pengaruh fitur di batch..."):
                bg_small = background.head(30)
                contrib_all = local_contributions(sample_b, bg_small)
            mean_abs = contrib_all.abs().mean()
            fig = plot_contribution_bar(mean_abs, top_n, "Rata-rata |kontribusi| fitur di batch")
            fig.update_traces(marker_color="#1f77b4", text=[f"{v:.2%}" for v in
                                                             mean_abs.sort_values(ascending=False)[:top_n].iloc[::-1]])
            st.plotly_chart(fig, width="stretch")
            if len(batch) > max_rows:
                st.caption(f"Dihitung dari sampel {max_rows} baris agar cepat.")


# =============================================================================
# TAB 3: INTERPRETASI MODEL (GLOBAL)
# =============================================================================
with tab_global:
    st.subheader("Faktor paling berpengaruh secara global")
    st.markdown(
        "**Permutation importance**: setiap fitur diacak satu per satu, lalu diukur seberapa besar "
        "ROC-AUC model turun. Semakin besar penurunannya, semakin penting fitur tersebut bagi model."
    )
    with st.spinner("Menghitung permutation importance..."):
        imp, base_auc = permutation_importance(ref, _model_key=model_key)
    if imp is None:
        st.info(f"Data referensi tidak memiliki kolom `{TARGET_COL}`, permutation importance tidak bisa dihitung.")
    else:
        st.metric("ROC-AUC model pada data referensi (sampel)", f"{base_auc:.3f}")
        top_imp = imp.head(top_n).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top_imp["importance"], y=top_imp["label"], orientation="h",
            error_x=dict(type="data", array=top_imp["std"]), marker_color="#1f77b4",
        ))
        fig.update_layout(title="Permutation importance (penurunan ROC-AUC)",
                          xaxis_title="Penurunan AUC", height=60 + 35 * len(top_imp))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Tabel lengkap"):
            st.dataframe(imp[["label", "feature", "importance", "std"]], width="stretch")

    st.markdown("---")
    st.subheader("Bagaimana satu fitur memengaruhi prediksi?")
    st.markdown(
        "**Partial dependence**: rata-rata probabilitas risiko ketika fitur terpilih di-set ke nilai "
        "tertentu, sementara fitur lain dibiarkan seperti aslinya."
    )
    default_feat = imp["feature"].iloc[0] if imp is not None else "AMT_INCOME_TOTAL"
    pd_feat = st.selectbox("Pilih fitur", FEATURE_COLS, index=FEATURE_COLS.index(default_feat),
                           format_func=label)
    with st.spinner("Menghitung partial dependence..."):
        pdp = partial_dependence(ref, pd_feat, _model_key=model_key)
    if pd_feat in CAT_COLS:
        pdp = pdp.sort_values("avg_proba")
        fig = px.bar(pdp, x="avg_proba", y="value", orientation="h")
        fig.update_layout(xaxis_tickformat=".2%", yaxis_title=label(pd_feat), xaxis_title="Rata-rata probabilitas")
    else:
        x = pdp["value"]
        x_title = label(pd_feat)
        if pd_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
            x = x / 365.25
            x_title = x_title.replace("(hari)", "(tahun)")
        fig = px.line(x=x, y=pdp["avg_proba"], markers=True)
        fig.update_layout(yaxis_tickformat=".2%", xaxis_title=x_title, yaxis_title="Rata-rata probabilitas")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Semua metode interpretasi di app ini model-agnostic: hanya memakai `predict_proba`, "
        "jadi tetap berlaku untuk model apa pun yang Anda pasang di bagian placeholder."
    )
