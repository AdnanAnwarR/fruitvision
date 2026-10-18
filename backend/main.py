import io, os, json, glob, base64
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

import database as db
from preprocess import smart_crop

MODEL_PATH   = os.getenv("MODEL_PATH", "models/fruitvision_effnetb0.keras")   # satu model (cadangan kalau tidak ada ensemble)
ENSEMBLE_DIR = os.getenv("ENSEMBLE_DIR", "models/ensemble")   # folder berisi beberapa model .keras (ensemble), dari notebook v3
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/class_names.json")
MULT_PATH    = os.getenv("MULT_PATH", "models/class_multipliers.json")   # dari notebook (cell 10)
# Cadangan kalau class_multipliers.json tidak ada: pengali Formalin-mixed manual (1.0 = normal)
FORMALIN_MULT = float(os.getenv("FORMALIN_MULT", "1.0"))
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/inference_config.json")   # dari notebook (cell 11)
USE_CROP = os.getenv("USE_CROP", "1") == "1"   # cadangan; dipakai kalau inference_config.json tidak ada
USE_TTA  = os.getenv("USE_TTA", "1") == "1"
MAX_BYTES = 10 * 1024 * 1024

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    paths = sorted(glob.glob(os.path.join(ENSEMBLE_DIR, "*.keras"))) or ([MODEL_PATH] if os.path.exists(MODEL_PATH) else [])
    if paths:
        import tensorflow as tf
        state["tf"] = tf
        state["models"] = [tf.keras.models.load_model(p) for p in paths]   # 1 model = biasa, >1 = ensemble
        state["size"] = tuple(state["models"][0].input_shape[1:3])   # otomatis 224 (B0) / 240 (B1)
        if os.path.exists(CLASSES_PATH):
            state["classes"] = json.load(open(CLASSES_PATH))
        else:
            state["classes"] = ["Fresh", "Rotten", "Formalin-mixed"]
        if os.path.exists(MULT_PATH):
            state["mult"] = json.load(open(MULT_PATH))
        else:
            state["mult"] = {"Formalin-mixed": FORMALIN_MULT}
        if os.path.exists(CONFIG_PATH):                 # pastikan sama persis dengan saat training
            cfg = json.load(open(CONFIG_PATH))
            state["use_crop"], state["use_tta"] = cfg.get("use_crop", True), cfg.get("use_tta", True)
        else:
            state["use_crop"], state["use_tta"] = USE_CROP, USE_TTA
        print("Model dimuat:", len(paths), "model | input:", state["size"])
    else:
        print(f"PERINGATAN: model tidak ditemukan di {ENSEMBLE_DIR} maupun {MODEL_PATH}")
    yield

app = FastAPI(title="FruitVision API (Apple Edition)", lifespan=lifespan)


def gradcam(model, x, k):
    """Heatmap Grad-CAM: bagian foto yang paling memengaruhi skor kelas k (0-1)."""
    tf = state["tf"]
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "models" in state, "n_models": len(state.get("models", []))}


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = False):
    if "models" not in state:
        raise HTTPException(503, "Model belum dimuat. Taruh file model di backend/models/")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File harus berupa gambar (jpg/png).")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ukuran gambar maksimal 10 MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if state["use_crop"]:
            img = smart_crop(img)          # sama persis seperti saat training
    except Exception:
        raise HTTPException(400, "Gambar tidak bisa dibaca.")

    tf, classes = state["tf"], state["classes"]
    arr = tf.image.resize(np.array(img, dtype="float32"), state["size"]).numpy()[None]  # tetap 0-255
    views = [arr] + ([arr[:, :, ::-1], arr[:, ::-1]] if state["use_tta"] else [])       # TTA: asli + flip
    probs = np.mean([m.predict(np.concatenate(views), verbose=0).mean(axis=0) for m in state["models"]], axis=0)   # rata-rata semua model (ensemble)

    adj = probs * np.array([state["mult"].get(c, 1.0) for c in classes])
    idx = int(adj.argmax())

    probs_dict = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    label, conf = classes[idx], float(probs[idx])
    pred_id = db.add_prediction(file.filename, label, conf, probs_dict)
    out = {"id": pred_id, "label": label, "confidence": round(conf, 4), "probs": probs_dict}
    if explain:                                                              # Grad-CAM dari model pertama
        cam = gradcam(state["models"][0], arr, idx)
        out["model_input_png"] = to_b64(img.resize((384, 384)))              # foto yang dilihat model (hasil crop)
        out["gradcam_png"] = to_b64(overlay(img.resize((384, 384)), cam))    # heatmap di atas foto
    return out


@app.get("/history")
def history(limit: int = 50):
    return db.list_predictions(limit)


@app.delete("/history/{pred_id}")
def delete_history(pred_id: int):
    if not db.delete_prediction(pred_id):
        raise HTTPException(404, "Data tidak ditemukan.")
    return {"deleted": pred_id}


@app.get("/stats")
def stats():
    return db.stats()
