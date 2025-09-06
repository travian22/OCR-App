# app.py — OCR Tabel → Tanggal×Bulan dengan PaddleOCR (fix Cloud)
import io, os, re, tempfile
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

# ========= Konstanta Bulan =========
MONTHS_STD = ["Jan","Peb","Mar","Apr","Mei","Jun","Jul","Ags","Sep","Okt","Nop","Des"]
MONTH_MAP = {
    "jan":"Jan","jan.":"Jan",
    "peb":"Peb","feb":"Peb","februari":"Peb",
    "mar":"Mar",
    "apr":"Apr",
    "mei":"Mei","may":"Mei",
    "jun":"Jun","juni":"Jun",
    "jul":"Jul","juli":"Jul",
    "ags":"Ags","agt":"Ags","aug":"Ags",
    "sep":"Sep","sept":"Sep",
    "okt":"Okt","oct":"Okt",
    "nop":"Nop","nov":"Nop","november":"Nop",
    "des":"Des","dec":"Des"
}

# ========= Helper Import PPStructure =========
def _get_PPStructure():
    try:
        from paddleocr.ppstructure import PPStructure  # versi stable (2.7.0.3)
        return PPStructure
    except Exception:
        from paddleocr import PPStructure
        return PPStructure

# ========= Preprocessing =========
def deskew(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thr, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            deg = theta * 180 / np.pi
            if deg < 15 or deg > 165:
                continue
            angles.append(deg - 90)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def upscale(img_bgr, scale=1.6):
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def preprocess_for_table(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 41, 9)
    return thr

def find_table_crops(img_bin):
    vertical = img_bin.copy()
    scale_v = max(10, img_bin.shape[1] // 120)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v))
    vertical = cv2.erode(vertical, kernel_v, iterations=1)
    vertical = cv2.dilate(vertical, kernel_v, iterations=1)

    horizontal = img_bin.copy()
    scale_h = max(10, img_bin.shape[0] // 120)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h, 1))
    horizontal = cv2.erode(horizontal, kernel_h, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel_h, iterations=1)

    grid = cv2.bitwise_and(vertical, horizontal)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 10000:
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1]//50, b[0]))
    return boxes

# ========= Parsing & Reshape =========
def to_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try: return float(s)
    except: return np.nan

def standardize_month(name):
    key = str(name).strip().lower().replace(" ", "").replace("-", "")
    return MONTH_MAP.get(key)

def reshape_from_month_row(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Tanggal"] + MONTHS_STD)

    df = df_raw.copy()
    df = df[~df.iloc[:,0].astype(str).str.lower().str.contains("rata")]

    months = []
    for v in df.iloc[:,0]:
        std = standardize_month(v)
        months.append(std if std else str(v))
    df.index = months

    df = df.drop(df.columns[0], axis=1)
    df.columns = pd.to_numeric(df.columns, errors="coerce")

    df_t = df.T
    df_t.index.name = "Tanggal"

    for c in df_t.columns:
        df_t[c] = df_t[c].map(to_float)

    for m in MONTHS_STD:
        if m not in df_t.columns:
            df_t[m] = np.nan
    df_t = df_t[MONTHS_STD]

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

# ========= Cache Engine & OCR =========
@st.cache_resource
def get_table_engine():
    PPStructure = _get_PPStructure()
    return PPStructure(layout=True, ocr=True, show_log=False)

@st.cache_data(show_spinner=False)
def ocr_table_bytes(img_bytes: bytes):
    engine = get_table_engine()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_bgr = deskew(img_bgr)
    img_bgr = upscale(img_bgr, 1.6)
    bin_img = preprocess_for_table(img_bgr)

    tmpdir = tempfile.mkdtemp()
    full_path = os.path.join(tmpdir, "full.png")
    cv2.imwrite(full_path, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    res = engine(full_path)
    tables = [r for r in res if r.get("type") == "table"]

    parsed = []
    def parse_items(items):
        for t in items:
            html = t.get("res",{}).get("html") or t.get("html")
            if not html: continue
            try:
                lst = pd.read_html(html)
                if lst: parsed.append(lst[0])
            except Exception: pass

    parse_items(tables)

    if not parsed:
        crops = find_table_crops(bin_img)
        for x,y,w,h in crops:
            crop_rgb = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            c_path = os.path.join(tmpdir, f"crop_{x}_{y}.png")
            cv2.imwrite(c_path, crop_rgb)
            sub = engine(c_path)
            sub_tables = [r for r in sub if r.get("type") == "table"]
            parse_items(sub_tables)

    return parsed

# ========= Streamlit UI =========
st.set_page_config(page_title="OCR Tabel → Tanggal×Bulan", layout="wide")
st.title("OCR Tabel → Tanggal × Bulan (PaddleOCR + Cache)")

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
        st.warning("Belum ada tabel yang bisa ditranspos.")
