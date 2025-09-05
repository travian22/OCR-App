# Deteksi Teks & Ekstraksi Tabel dari Gambar

Aplikasi ini menggunakan Streamlit untuk mendeteksi dan mengekstrak tabel dari gambar (format .png, .jpg, .jpeg) menggunakan PaddleOCR PP-Structure. Hasil ekstraksi ditampilkan dalam bentuk tabel mentah dan versi transposisi (baris = tanggal, kolom = bulan).

## Fitur
- Upload gambar tabel (JPG/PNG)
- Deteksi otomatis tabel pada gambar
- Ekstraksi data tabel menggunakan PaddleOCR
- Tampilkan tabel mentah dan hasil transposisi (Tanggal × Bulan)
- Mendukung berbagai format nama bulan dalam bahasa Indonesia

## Cara Menjalankan
1. **Clone/download** repositori ini dan masuk ke foldernya.
2. (Opsional) Buat dan aktifkan virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi Streamlit:**
   ```powershell
   streamlit run app.py
   ```
5. Buka browser ke alamat yang tertera (biasanya http://localhost:8501)

## Struktur File
- `app.py` — Kode utama aplikasi Streamlit
- `requirements.txt` — Daftar dependensi Python

## Catatan
- Aplikasi ini membutuhkan PaddleOCR dan PaddlePaddle (CPU). Pastikan sudah terinstall sesuai `requirements.txt`.
- Jika terjadi error terkait tampilan tabel, gunakan versi Streamlit terbaru.
- Untuk hasil optimal, gunakan gambar tabel yang jelas dan rapi.

## Lisensi
Aplikasi ini bebas digunakan untuk keperluan non-komersial dan edukasi.
