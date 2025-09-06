# app.py — OCR Tabel → Tanggal×Bulan (PaddleOCR, tanpa OpenCV)
import io, os, re, tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ==== Konstanta Bulan ====
MONTHS_STD = ["Jan","Peb","Mar","Apr","Mei","Jun","Jul","Ags","Sep","Okt","Nop","Des"]
MONTH_MAP = {
    "jan":"Jan","jan.":"Jan",
    "peb":"Peb","feb":"Peb","februari":"Peb",
    "mar":"Mar","apr":"Apr",
    "mei":"Mei","may":"Mei",
    "jun":"Jun","juni":"Jun",
    "jul":"Jul","juli":"Jul",
    "ags":"Ags","agt":"Ags","aug":"Ags",
    "sep":"Sep","sept":"Sep",
    "okt":"Okt","oct":"Okt",
    "nop":"Nop","nov":"Nop","november":"Nop",
    "des":"Des","dec":"Des"
}

# ==== Import PPStructure (kompatibel lintas versi) ====
def _get_PPStructure():
    try:
        from paddleocr.ppstructure import PPStructure  # prefer di 2.7.0.3
        return PPStructure
    except Exception:
        from paddleocr import PPStructure
        return PPStructure

# ==== Util tanpa OpenCV ====
def pil_max_side(img_pil: Image.Image, max_side=2200) -> Image.Image:
    w, h = img_pil.size
    m = max(w, h)
    if m <= max_side:
        return img_pil
    scale = max_side / float(m)
    return img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def to_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try: return float(s)
    except: return np.nan

def standardize_month(name):
    key = str(name).strip().lower().replace(" ", "").replace("-", "")
    return MONTH_MAP.get(key)

# ==== Reshape: baris bulan → baris tanggal ====
def reshape_from_month_row(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Input OCR: baris = bulan, kolom = tanggal
    # Output: baris = tanggal (1..31), kolom = Jan..Des
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Tanggal"] + MONTHS_STD)

    df = df_raw.copy()
    first_col = df.columns[0]

    # buang baris 'rata'
    df = df[~df[first_col].astype(str).str.lower().str.contains("rata")]

    # kolom pertama = label bulan
    months = []
    for v in df[first_col].tolist():
        std = standardize_month(v)
        months.append(std if std else str(v))
    df.index = months
    df.drop(columns=[first_col], inplace=True)

    # header tanggal → numerik
    df.columns = pd.to_numeric(df.columns, errors="coerce")

    # transpose → baris = tanggal, kolom = bulan
    df_t = df.T
    df_t.index.name = "Tanggal"

    # nilai ke float
    for c in df_t.columns:
        df_t[c] = df_t[c].map(to_float)

    # pastikan semua bulan ada & urut
    for m in MONTHS_STD:
        if m not in df_t.columns:
            df_t[m] = np.nan
    df_t = df_t[MONTHS_STD]

    # jadikan tanggal kolom biasa + lengkapi 1..31
    df_t = df_t.reset_index()
    idx_full = pd.DataFrame({"Tanggal": list(range(1,32))})
    df_t = idx_full.merge(df_t, on="Tanggal", how="left")
    return df_t

def merge_tanggal_bulan_tables(dfs):
    if not dfs:
        return pd.DataFrame(columns=["Tanggal"] + MONTHS_STD)
    merged = dfs[0].copy()
    for nxt in dfs[1:]:
        merged = merged.merge(nxt, on="Tanggal", how="outer", suffixes=("","_dup"))
        for m in MONTHS_STD:
            dup = f"{m}_dup"
            if dup in merged.columns:
                merged[m] = merged[m].combine_first(merged[dup])
                merged.drop(columns=[dup], inplace=True)
    merged = merged.sort_values("Tanggal")
    merged = merged[merged["Tanggal"].between(1,31)]
    return merged[["Tanggal"] + MONTHS_STD].reset_index(drop=True)

# ==== Cache model & OCR ====
@st.cache_resource
def get_table_engine():
    PPStructure = _get_PPStructure()
    return PPStructure(layout=True, ocr=True, show_log=False)

@st.cache_data(show_spinner=False)
def ocr_table_bytes(img_bytes: bytes):
    # Jalankan PP-Structure pada gambar (tanpa OpenCV)
    engine = get_table_engine()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = pil_max_side(img, 2200)

    # simpan sementara
    tmpdir = tempfile.mkdtemp()
    full_path = os.path.join(tmpdir, "full.png")
    img.save(full_path, format="PNG")

    res = engine(full_path)
    tables = [r for r in res if r.get("type") == "table"]

    parsed = []
    def parse_items(items):
        for t in items:
            html = (t.get("res", {}) or {}).get("html") or t.get("html")
            if not html:
                continue
            try:
                lst = pd.read_html(html)
                if lst:
                    parsed.append(lst[0])
            except Exception:
                pass

    parse_items(tables)
    return parsed

# ==== Streamlit UI ====
st.set_page_config(page_title="OCR Tabel → Tanggal×Bulan", layout="wide")
st.title("OCR Tabel → Tanggal × Bulan (PaddleOCR, Cloud-safe)")

uploaded = st.file_uploader("Upload gambar tabel (.png/.jpg/.jpeg)", type=["png","jpg","jpeg"])
if uploaded:
    uploaded.seek(0)
    img_bytes = uploaded.read()
    st.image(Image.open(io.BytesIO(img_bytes)), caption="Gambar Masukan")

    with st.spinner("Mendeteksi & mengekstrak tabel..."):
        raw_tables = ocr_table_bytes(img_bytes)

    st.success(f"Terdeteksi {len(raw_tables)} tabel.")

    reshaped = []
    for i, tdf in enumerate(raw_tables, 1):
        st.subheader(f"Tabel Mentah #{i}")
        st.dataframe(tdf)

        df_tb = reshape_from_month_row(tdf)
        if not df_tb.empty:
            reshaped.append(df_tb)

    if reshaped:
        final = merge_tanggal_bulan_tables(reshaped)
        st.subheader("Tabel hasil transpos (Tanggal × Bulan)")
        st.dataframe(final)

        # download excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf) as writer:
            final.to_excel(writer, index=False, sheet_name="Tanggal×Bulan")
        st.download_button(
            "Unduh Excel",
            data=buf.getvalue(),
            file_name="tanggal_x_bulan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Belum ada tabel yang bisa ditranspos. Coba gambar resolusi lebih tinggi atau crop area tabel.")
