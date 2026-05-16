import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

BASE_DIR = Path("datasets/train")
BENIGN_DIR = BASE_DIR / "benign"
MALIGNANT_DIR = BASE_DIR / "malignant"
NORMAL_DIR = BASE_DIR / "Normal"

OUT_DIR = Path("datasets/synthetic_yolo")
IMG_SIZE = 640
N_TRAIN = 4000  # cant imagenes entrenamiento
N_VAL = 500  # cant imagenes validacion
# N_TRAIN = 10 #para probar que funcione
# N_VAL = 5
VAL_BG_FRACTION = 0.15  # 15% de los fondos van solo a val (evita data leakage)
SINGLE_CLASS = True  # True: YOLO solo detecta "mole". False: separa benign/malignant
MIN_PATCHES = 1
MAX_PATCHES = 2


def sample_patch_size(img_size=IMG_SIZE):
    """Sesgado hacia tamaños chicos — refleja distribución real."""
    r = random.random()
    if r < 0.70:
        frac = random.uniform(0.04, 0.08)
    elif r < 0.95:
        frac = random.uniform(0.08, 0.18)
    else:
        frac = random.uniform(0.18, 0.25)
    return int(img_size * frac)


def make_soft_mask(h, w, feather=0.18):
    mask = np.ones((h, w), dtype=np.float32)
    fh, fw = int(h * feather), int(w * feather)
    for i in range(fh):
        alpha = i / fh
        mask[i, :] *= alpha
        mask[h - 1 - i, :] *= alpha
    for j in range(fw):
        alpha = j / fw
        mask[:, j] *= alpha
        mask[:, w - 1 - j] *= alpha
    return (mask * 255).astype(np.uint8)


def color_match(patch, bg_roi):
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    p_mean, p_std = patch_lab[..., 0].mean(), patch_lab[..., 0].std() + 1e-6
    b_mean, b_std = bg_lab[..., 0].mean(), bg_lab[..., 0].std() + 1e-6
    blend = 0.5
    new_mean = p_mean * (1 - blend) + b_mean * blend
    new_std = p_std * (1 - blend) + b_std * blend
    patch_lab[..., 0] = (patch_lab[..., 0] - p_mean) * (new_std / p_std) + new_mean
    patch_lab = np.clip(patch_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(patch_lab, cv2.COLOR_LAB2BGR)


def augment_patch(patch):
    h, w = patch.shape[:2]
    angle = random.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    patch = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.5:
        patch = cv2.flip(patch, 1)
    if random.random() < 0.3:
        patch = cv2.flip(patch, 0)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        patch = cv2.GaussianBlur(patch, (k, k), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 8), patch.shape)
        patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return patch


def paste_patch(bg, patch, x, y):
    ph, pw = patch.shape[:2]
    H, W = bg.shape[:2]
    x = max(0, min(x, W - pw))
    y = max(0, min(y, H - ph))
    bg_roi = bg[y : y + ph, x : x + pw].copy()
    patch_matched = color_match(patch, bg_roi)
    mask = make_soft_mask(ph, pw, feather=0.18)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    blended = bg_roi.astype(np.float32) * (1 - m) + patch_matched.astype(np.float32) * m
    bg[y : y + ph, x : x + pw] = blended.astype(np.uint8)
    cx = (x + pw / 2) / W
    cy = (y + ph / 2) / H
    bw = pw / W
    bh = ph / H
    return bg, (cx, cy, bw, bh)


def boxes_overlap(b1, b2, iou_thresh=0.05):
    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    x1a, y1a, x2a, y2a = to_xyxy(b1)
    x1b, y1b, x2b, y2b = to_xyxy(b2)
    inter = max(0, min(x2a, x2b) - max(x1a, x1b)) * max(
        0, min(y2a, y2b) - max(y1a, y1b)
    )
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union > iou_thresh if union > 0 else False


def load_images(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return files


def get_class_id(mole_path):
    """Devuelve el class_id según la carpeta de origen."""
    if SINGLE_CLASS:
        return 0
    return 0 if "benign" in mole_path.parent.name.lower() else 1


def generate_one(bg_path, mole_pool, out_img, out_lbl):
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return False
    bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

    n = random.randint(MIN_PATCHES, MAX_PATCHES)
    labels = []
    placed = []

    for _ in range(n * 5):
        if len(labels) >= n:
            break
        mole_path = random.choice(mole_pool)
        patch = cv2.imread(str(mole_path))
        if patch is None:
            continue

        target = sample_patch_size(IMG_SIZE)
        patch = cv2.resize(patch, (target, target), interpolation=cv2.INTER_AREA)
        patch = augment_patch(patch)

        x = random.randint(0, IMG_SIZE - target)
        y = random.randint(0, IMG_SIZE - target)
        candidate = (
            (x + target / 2) / IMG_SIZE,
            (y + target / 2) / IMG_SIZE,
            target / IMG_SIZE,
            target / IMG_SIZE,
        )
        if any(boxes_overlap(candidate, pb) for pb in placed):
            continue

        bg, box = paste_patch(bg, patch, x, y)
        placed.append(box)
        cls = get_class_id(mole_path)
        labels.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

    if not labels:
        return False
    cv2.imwrite(str(out_img), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_lbl.write_text("\n".join(labels))
    return True


def write_data_yaml():
    if SINGLE_CLASS:
        names_block = "names:\n  0: mole"
    else:
        names_block = "names:\n  0: benign\n  1: malignant"

    yaml_content = f"""# Auto-generado por generar_dataset.py
path: {OUT_DIR.resolve()}
train: images/train
val: images/val

{names_block}
"""
    (OUT_DIR / "data.yaml").write_text(yaml_content)
    print(f"  data.yaml escrito en {OUT_DIR / 'data.yaml'}")


def main():
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    benign = load_images(BENIGN_DIR)
    malignant = load_images(MALIGNANT_DIR)
    normal = load_images(NORMAL_DIR)

    assert benign or malignant, "No hay lunares en benign/ ni malignant/"
    assert normal, f"No hay fondos en {NORMAL_DIR}"

    moles = benign + malignant
    print(f"Lunares: {len(moles)} (benign={len(benign)}, malignant={len(malignant)})")
    print(f"Fondos:  {len(normal)}")

    # Split de fondos para evitar data leakage
    random.shuffle(normal)
    n_val_bg = max(1, int(len(normal) * VAL_BG_FRACTION))
    bgs_val = normal[:n_val_bg]
    bgs_train = normal[n_val_bg:]
    print(f"Fondos train: {len(bgs_train)} | Fondos val: {len(bgs_val)}")

    # Split de lunares tambien (mas conservador)
    random.shuffle(moles)
    n_val_moles = max(1, int(len(moles) * VAL_BG_FRACTION))
    moles_val = moles[:n_val_moles]
    moles_train = moles[n_val_moles:]

    splits = [
        ("train", N_TRAIN, bgs_train, moles_train),
        ("val", N_VAL, bgs_val, moles_val),
    ]

    for split_name, total, bg_pool, mole_pool in splits:
        success = 0
        pbar = tqdm(range(total), desc=f"Generando {split_name}")
        for i in pbar:
            bg_path = random.choice(bg_pool)
            out_img = OUT_DIR / "images" / split_name / f"{split_name}_{i:05d}.jpg"
            out_lbl = OUT_DIR / "labels" / split_name / f"{split_name}_{i:05d}.txt"
            ok = generate_one(bg_path, mole_pool, out_img, out_lbl)
            if not ok:
                ok = generate_one(random.choice(bg_pool), mole_pool, out_img, out_lbl)
            if ok:
                success += 1
        print(f"  {split_name}: {success}/{total} imágenes generadas")

    write_data_yaml()
    print("\n Dataset listo para entrenar YOLO.")


if __name__ == "__main__":
    main()
