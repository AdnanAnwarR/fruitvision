# FruitVision (Apple Edition) — End-to-End
Frontend: Streamlit · Backend: FastAPI · Database: SQLite · Model: EfficientNet (Keras)

## Struktur
```
fruitvision/
├── streamlit_app.py       # ★ versi mandiri: prediksi + Grad-CAM + riwayat dalam 1 proses Streamlit
├── requirements.txt       # dependensi untuk versi mandiri (Streamlit Cloud)
├── .python-version        # 3.12 — versi Python untuk Streamlit Cloud
├── backend/
│   ├── main.py            # REST API (FastAPI)
│   ├── database.py        # SQLite (riwayat prediksi)
│   ├── preprocess.py      # crop apel (HARUS sama dengan saat training)
│   ├── requirements.txt
│   └── models/            # <- taruh model hasil Colab di sini
└── frontend/
    ├── app.py             # UI Streamlit (versi lokal, memanggil backend FastAPI)
    └── requirements.txt
```

## 1. Model
Model hasil training **sudah termasuk** di `backend/models/`:
- `fruitvision_effnetb0.keras` (EfficientNet-B0, dilatih dengan semua data)
- `class_names.json` dan `inference_config.json` (crop dibaca otomatis sesuai training)

`preprocess.py` di `backend/` adalah file yang sama dengan yang dipakai saat training.
Untuk mengganti model, timpa file-file di atas dengan hasil ekspor notebook (`fruitvision_export.zip`).
Python 3.10 sampai 3.14 didukung (di 3.14 pakai TF ≥2.22 — sudah diatur otomatis di `requirements.txt`).

## Performa model (jujur)
Hasil 5-fold per blok ID pada 8.461 foto: **akurasi 0,78 (±0,10 antar fold), macro F1 0,72**.
Satu blok data (fold 5) jauh lebih sulit (akurasi 0,63). Formalin-mixed paling sulit dibedakan dari Fresh
(recall 0,72). Model ini alat bantu awal, **bukan pengganti uji laboratorium**.

## 2. Jalankan backend (terminal 1)
```
cd backend
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Dokumentasi API otomatis: http://localhost:8000/docs

Kalau pakai model B1: `set MODEL_PATH=models/fruitvision_effnetb1.keras` (Mac/Linux: `export ...`) sebelum uvicorn.

## 3. Jalankan frontend (terminal 2)
```
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501

## 4. Deploy ke Streamlit Community Cloud (gratis)
`streamlit_app.py` adalah versi mandiri: prediksi model, Grad-CAM, dan riwayat berjalan
langsung di dalam Streamlit, jadi **tidak perlu server FastAPI terpisah** — cukup satu deploy.

1. **Buat repo GitHub** (publik) di https://github.com/new — nama bebas (mis. `fruitvision`),
   **jangan** centang "Add a README file" (kita akan push dari lokal).
2. **Push projek** dari folder ini (jendela browser akan terbuka untuk login GitHub):
   ```
   git remote add origin https://github.com/<username>/fruitvision.git
   git branch -M main
   git push -u origin main
   ```
   File model (~43 MB) di-push biasa — masih di bawah batas 100 MB GitHub, jadi Git LFS tidak wajib.
3. **Deploy**: buka https://streamlit.io/cloud → *Sign in with GitHub* → *New app*, lalu isi:
   | Field | Isi dengan |
   |---|---|
   | Repository | `<username>/fruitvision` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` |
   | Advanced settings → Python version | **3.12** atau **3.14** — keduanya didukung (`requirements.txt` otomatis memilih TF yang cocok) |
4. Tunggu build 5–10 menit → selesai. App bisa diakses di `https://<nama-pilihan>.streamlit.app`.

Catatan untuk versi cloud:
- **Riwayat** disimpan di `/tmp` → hilang saat server restart (prediksi, Grad-CAM, statistik tetap jalan).
- Perubahan kode cukup `git push`; Streamlit Cloud deploy ulang otomatis.
- Arsitektur lokal (frontend + backend FastAPI) tetap bisa dipakai: ikuti langkah **2** dan **3** di atas.

## Fitur frontend
- Upload atau kamera, hasil kelas + keyakinan + grafik probabilitas.
- **Grad-CAM** (centang di tab Prediksi): menampilkan bagian foto yang diperhatikan model (merah = paling berpengaruh).
- Tab Riwayat: tabel prediksi, statistik per kelas, hapus riwayat.

## Endpoint
| Method | Path | Fungsi |
|---|---|---|
| GET | /health | cek status & model |
| POST | /predict?explain=true | upload gambar, hasil prediksi + simpan ke DB; `explain=true` menambah gambar Grad-CAM |
| GET | /history?limit=50 | riwayat prediksi |
| DELETE | /history/{id} | hapus satu riwayat |
| GET | /stats | jumlah prediksi per kelas |

## Catatan penting
- **Crop & TTA**: notebook v2/v3 melatih model dengan gambar yang di-crop ke apel (`USE_CROP=True`), jadi backend juga meng-crop
  gambar upload dengan `preprocess.py`. Kalau `inference_config.json` ada, backend otomatis mengikutinya; kalau tidak, atur lewat `USE_CROP=0/1` dan `USE_TTA=0/1`.
- **Pengali probabilitas**: dibaca dari `models/class_multipliers.json`. Kalau file tidak ada, pakai `FORMALIN_MULT` (default 1.0).
