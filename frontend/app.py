import os
import base64
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FruitVision - Apple Edition", page_icon="🍎", layout="centered")

API_URL = st.sidebar.text_input("URL Backend (FastAPI)", os.getenv("API_URL", "http://localhost:8000"))
EMOJI = {"Fresh": "✅", "Rotten": "🟤", "Formalin-mixed": "⚠️"}

st.title("🍎 FruitVision — Apple Edition")
st.caption("Deteksi kondisi apel: Fresh / Rotten / Formalin-mixed")

try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    if h.get("model_loaded"):
        st.sidebar.success("Backend aktif, model dimuat")
    else:
        st.sidebar.warning("Backend aktif, tapi model belum dimuat")
except Exception:
    st.sidebar.error("Backend tidak terjangkau")

st.sidebar.caption("Performa model (5-fold per blok ID): akurasi ±78% (±10), macro F1 0,72. "
                   "Formalin-mixed paling sulit dibedakan dari Fresh. Ini alat bantu awal, "
                   "bukan pengganti uji laboratorium.")

tab1, tab2 = st.tabs(["🔍 Prediksi", "🗂️ Riwayat"])

with tab1:
    mode = st.radio("Sumber gambar", ["Upload", "Kamera"], horizontal=True)
    img = st.file_uploader("Pilih foto apel", type=["jpg", "jpeg", "png"]) if mode == "Upload" \
        else st.camera_input("Ambil foto apel")

    show_cam = st.checkbox("Tampilkan Grad-CAM (bagian foto yang diperhatikan model)", value=True)

    if img is not None:
        st.image(img, width=300)
        if st.button("Prediksi", type="primary"):
            with st.spinner("Menganalisis..."):
                try:
                    r = requests.post(f"{API_URL}/predict",
                                      files={"file": (img.name, img.getvalue(), img.type)},
                                      params={"explain": str(show_cam).lower()}, timeout=120)
                    if r.status_code != 200:
                        st.error(r.json().get("detail", "Terjadi kesalahan"))
                    else:
                        res = r.json()
                        st.subheader(f"{EMOJI.get(res['label'], '')} {res['label']}")
                        st.metric("Keyakinan model", f"{res['confidence']:.1%}")
                        st.bar_chart(pd.Series(res["probs"], name="Probabilitas"))
                        if res["label"] == "Formalin-mixed":
                            st.warning("Dugaan formalin dari foto bersifat awal. Konfirmasi dengan uji laboratorium.")
                        if res.get("gradcam_png"):
                            c1, c2 = st.columns(2)
                            c1.image(base64.b64decode(res["model_input_png"]), caption="Yang dilihat model (hasil crop)")
                            c2.image(base64.b64decode(res["gradcam_png"]), caption="Grad-CAM: merah = paling berpengaruh")
                            st.caption("Kalau area merah jatuh di apel (terutama bagian busuk atau tekstur kulit), model melihat hal yang tepat. "
                                       "Kalau jatuh di latar, hasil prediksi perlu diragukan.")
                        if res["confidence"] < 0.6:
                            st.info("Keyakinan rendah, hasil sebaiknya dicek ulang (foto lebih jelas / pencahayaan lebih baik).")
                except Exception as e:
                    st.error(f"Gagal terhubung ke backend: {e}")

with tab2:
    if st.button("Muat ulang"):
        st.rerun()
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        if stats:
            st.bar_chart(pd.Series(stats, name="Jumlah prediksi"))
        rows = requests.get(f"{API_URL}/history", params={"limit": 50}, timeout=5).json()
        if rows:
            df = pd.DataFrame(rows)[["id", "created_at", "filename", "label", "confidence"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
            del_id = st.number_input("Hapus riwayat (ID)", min_value=0, step=1)
            if st.button("Hapus") and del_id:
                d = requests.delete(f"{API_URL}/history/{int(del_id)}", timeout=5)
                st.success("Terhapus") if d.status_code == 200 else st.error("ID tidak ditemukan")
        else:
            st.info("Belum ada riwayat prediksi.")
    except Exception:
        st.error("Tidak bisa mengambil riwayat dari backend.")
