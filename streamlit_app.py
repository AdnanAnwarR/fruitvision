# streamlit_app.py — FruitVision (Apple Edition), versi mandiri untuk Streamlit Community Cloud.
#
# Di sini prediksi berjalan langsung di dalam aplikasi Streamlit (tanpa server FastAPI terpisah),
# sehingga projek cukup di-deploy sekali ke https://streamlit.io/cloud.
# Arsitektur lokal (frontend Streamlit + backend FastAPI) tetap tersedia di frontend/ dan backend/.
# Petunjuk deploy ada di README.md bagian "4. Deploy ke Streamlit Community Cloud".

import os
import io
import sys
import json
import glob
import base64
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="FruitVision - Apple Edition", page_icon="🍎", layout="centered")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # supaya backend/preprocess.py (crop dari training) dan backend/database.py bisa diimpor

# Riwayat prediksi disimpan di folder sementara: di Streamlit Cloud hanya /tmp yang bisa ditulis,
# dan datanya akan ter-reset saat server aplikasi dimuat ulang.
_TMP = "/tmp" if os.access("/tmp", os.W_OK) else tempfile.gettempdir()
os.environ.setdefault("DB_PATH", os.path.join(_TMP, "fruitvision.db"))  # harus diatur sebelum backend.database dibaca

from backend import database as db
from backend.preprocess import smart_crop

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT, "backend", "models"))
EMOJI = {"Fresh": "✅", "Rotten": "🟤", "Formalin-mixed": "⚠️"}


def _read_json(path, default):
    """Baca file JSON; kalau tidak ada atau rusak, pakai nilai default."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


@st.cache_resource(show_spinner="Memuat model EfficientNet...")
def load_model():
    """Muat model (.keras), daftar kelas, pengali, dan konfigurasi inference — sekali per server."""
    model_path = os.path.join(MODEL_DIR, "fruitvision_effnetb0.keras")
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "ensemble", "*.keras"))) or \
        ([model_path] if os.path.exists(model_path) else [])
    if not paths:
        return None
    import tensorflow as tf
    models = [tf.keras.models.load_model(p) for p in paths]   # 1 model = biasa, >1 = ensemble

    cfg = _read_json(os.path.join(MODEL_DIR, "inference_config.json"), None)
    return {
        "tf": tf,
        "models": models,
        "size": tuple(models[0].input_shape[1:3]),   # otomatis 224 (B0) / 240 (B1)
        "classes": _read_json(os.path.join(MODEL_DIR, "class_names.json"), ["Fresh", "Rotten", "Formalin-mixed"]),
        "mult": _read_json(os.path.join(MODEL_DIR, "class_multipliers.json"),
                           {"Formalin-mixed": float(os.getenv("FORMALIN_MULT", "1.0"))}),
        # crop & TTA harus sama persis dengan saat training (dibaca dari inference_config.json)
        "use_crop": cfg.get("use_crop", True) if cfg else os.getenv("USE_CROP", "1") == "1",
        "use_tta": cfg.get("use_tta", True) if cfg else os.getenv("USE_TTA", "1") == "1",
    }


def gradcam(tf, model, x, k):
    """Heatmap Grad-CAM: bagian foto yang paling memengaruhi skor kelas k (0-1)."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))    # bagian EfficientNet
    post = model.layers[model.layers.index(base) + 1:]                       # lapisan setelah EfficientNet
    with tf.GradientTape() as tape:
        conv = base(tf.convert_to_tensor(x), training=False)                 # peta fitur terakhir
        tape.watch(conv)
        y = conv
        for l in post:
            y = l(y, training=False)
        score = y[:, k]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, cam, alpha=0.45):
    """Tempel heatmap (skala warna jet: biru = rendah, merah = tinggi) di atas foto."""
    h = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)) / 255.0
    rgb = np.stack([np.clip(1.5 - np.abs(4 * h - 3), 0, 1),
                    np.clip(1.5 - np.abs(4 * h - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * h - 1), 0, 1)], axis=-1)
    return Image.blend(img.convert("RGB"), Image.fromarray((rgb * 255).astype("uint8")), alpha)


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict_image(bundle, data, filename, explain):
    """Inference persis seperti backend FastAPI: crop -> resize -> (TTA) -> rata-rata model -> pengali -> simpan riwayat."""
    tf = bundle["tf"]
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if bundle["use_crop"]:
        img = smart_crop(img)          # sama persis seperti saat training
    arr = tf.image.resize(np.array(img, dtype="float32"), bundle["size"]).numpy()[None]  # tetap 0-255
    views = [arr] + ([arr[:, :, ::-1], arr[:, ::-1]] if bundle["use_tta"] else [])       # TTA: asli + flip
    probs = np.mean([m.predict(np.concatenate(views), verbose=0).mean(axis=0) for m in bundle["models"]], axis=0)

    adj = probs * np.array([bundle["mult"].get(c, 1.0) for c in bundle["classes"]])
    idx = int(adj.argmax())
    probs_dict = {c: round(float(p), 4) for c, p in zip(bundle["classes"], probs)}
    label, conf = bundle["classes"][idx], float(probs[idx])
    db.add_prediction(filename, label, conf, probs_dict)  # riwayat disimpan di SQLite sementara

    res = {"label": label, "confidence": round(conf, 4), "probs": probs_dict}
    if explain:                                                              # Grad-CAM dari model pertama
        cam = gradcam(tf, bundle["models"][0], arr, idx)
        res["model_input_png"] = to_b64(img.resize((384, 384)))              # foto yang dilihat model (hasil crop)
        res["gradcam_png"] = to_b64(overlay(img.resize((384, 384)), cam))    # heatmap di atas foto
    return res


# ------------------------------- Antarmuka aplikasi -------------------------------

db.init_db()
bundle = load_model()

st.title("🍎 FruitVision — Apple Edition")
st.caption("Deteksi kondisi apel: Fresh / Rotten / Formalin-mixed")

if bundle is None:
    st.sidebar.error("Model tidak ditemukan — pastikan file backend/models/*.keras ikut ada di repo GitHub")
else:
    st.sidebar.success(f"Model siap — {len(bundle['models'])} EfficientNet, "
                       f"crop={'aktif' if bundle['use_crop'] else 'nonaktif'}, "
                       f"TTA={'aktif' if bundle['use_tta'] else 'nonaktif'}")

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
            if bundle is None:
                st.error("Model belum dimuat. Pastikan file backend/models/*.keras ikut ter-upload ke GitHub.")
            else:
                with st.spinner("Menganalisis..."):
                    try:
                        res = predict_image(bundle, img.getvalue(), img.name, explain=show_cam)
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
                        st.error(f"Gagal memproses gambar: {e}")

with tab2:
    st.caption("Riwayat disimpan sementara di server; bisa ter-reset saat aplikasi dimuat ulang.")
    if st.button("Muat ulang"):
        st.rerun()
    stats = db.stats()
    if stats:
        st.bar_chart(pd.Series(stats, name="Jumlah prediksi"))
    rows = db.list_predictions(50)
    if rows:
        df = pd.DataFrame(rows)[["id", "created_at", "filename", "label", "confidence"]]
        st.dataframe(df, hide_index=True)
        del_id = st.number_input("Hapus riwayat (ID)", min_value=0, step=1)
        if st.button("Hapus") and del_id:
            st.success("Terhapus") if db.delete_prediction(int(del_id)) else st.error("ID tidak ditemukan")
    else:
        st.info("Belum ada riwayat prediksi.")
