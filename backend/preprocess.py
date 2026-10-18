# preprocess.py — dipakai SAMA PERSIS saat training (notebook) dan inference (backend)
import numpy as np                      # numpy: hitungan array/angka
from PIL import Image                   # PIL: membuka dan mengolah gambar

OUT = 256                               # ukuran sisi gambar hasil crop (nanti di-resize lagi ke 224/240)


def smart_crop(img, out=OUT, margin=0.10):
    """Crop persegi di sekitar apel (deteksi lewat warna jenuh) dan tutup sudut hitam bekas rotasi."""
    img = img.convert("RGB")            # pastikan gambar berformat RGB (3 warna)
    w, h = img.size                     # lebar dan tinggi gambar asli
    s = min(1.0, 512 / max(w, h))       # faktor kecil: gambar terlalu besar dikecilkan agar cepat
    if s < 1.0:                         # kalau memang perlu dikecilkan
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))   # kecilkan gambar
    a = np.asarray(img).astype("float32")   # ubah gambar menjadi array angka
    H, W, _ = a.shape                   # tinggi dan lebar array

    valid = a.max(axis=2) > 25          # True = piksel bukan hitam; False = piksel hitam (sudut rotasi)
    if valid.sum() < 0.2 * H * W:       # kalau hampir semua gambar gelap
        valid[:] = True                 # anggap semua piksel valid

    b = max(2, int(0.06 * min(H, W)))   # tebal pinggiran (6% dari sisi terpendek)
    edge = np.zeros((H, W), bool)       # peta pinggiran, awalnya semua False
    edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True   # tandai pinggiran atas, bawah, kiri, kanan
    ref = a[edge & valid] if (edge & valid).sum() > 20 else a[valid]  # piksel pinggiran yang bukan hitam = contoh warna latar
    a[~valid] = np.median(ref, axis=0)  # tutup piksel hitam dengan warna latar (nilai tengah)

    sat = a.max(axis=2) - a.min(axis=2) # saturasi: selisih warna terkuat dan terlemah (latar abu-abu = kecil)
    mask = (sat > 45) & valid           # apel = piksel yang berwarna (jenuh)
    rows = np.where(mask.mean(axis=1) > 0.03)[0]   # baris yang mengandung apel
    cols = np.where(mask.mean(axis=0) > 0.03)[0]   # kolom yang mengandung apel
    if len(rows) and len(cols):         # kalau apel terdeteksi
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1   # batas atas, bawah, kiri, kanan apel
    else:                               # kalau tidak terdeteksi
        y0 = x0 = y1 = x1 = 0           # beri nilai nol (nanti memicu crop tengah)
    if (y1 - y0) * (x1 - x0) < 0.08 * H * W:        # kalau area apel terlalu kecil
        y0, y1, x0, x1 = int(.1 * H), int(.9 * H), int(.1 * W), int(.9 * W)   # pakai crop tengah (80% gambar)

    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))   # sisi kotak crop = sisi apel terbesar + margin
    top = int((y0 + y1) / 2 - side / 2)  # posisi atas kotak (apel di tengah)
    left = int((x0 + x1) / 2 - side / 2) # posisi kiri kotak
    p = side                            # lebar bantalan pinggir (supaya crop tidak keluar gambar)
    ap = np.pad(a, ((p, p), (p, p), (0, 0)), mode="edge")  # tambah pinggiran dengan menyalin piksel tepi
    crop = ap[top + p: top + p + side, left + p: left + p + side]   # potong kotak di sekitar apel
    return Image.fromarray(crop.astype("uint8")).resize((out, out), Image.BILINEAR)   # ubah ke gambar 256x256
