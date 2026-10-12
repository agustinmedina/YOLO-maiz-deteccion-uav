#!/usr/bin/env python3
"""Convierte el dataset 'Plant Detection and Counting' (formato Supervisely) a formato YOLO,
dividiendo las imagenes grandes en tiles y separando train/val/test por GRUPO (sitio/vuelo).

Uso:
    python supervisely_to_yolo.py RUTA_DATASET SALIDA [--tile 1024] [--overlap 128]
                                  [--group-regex REGEX] [--val 0.15] [--test 0.15] [--seed 42]

RUTA_DATASET: carpeta del proyecto Supervisely (con meta.json y subcarpetas <ds>/img y <ds>/ann).
"""
import argparse, json, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def find_pairs(root: Path):
    """Devuelve lista de (ruta_imagen, ruta_anotacion) recorriendo todos los datasets del proyecto."""
    pairs = []
    for ann in sorted(root.rglob("ann/*.json")):
        img_dir = ann.parent.parent / "img"
        img = img_dir / ann.name[:-5]  # quita '.json' -> nombre original de la imagen
        if img.exists():
            pairs.append((img, ann))
        else:
            print(f"[aviso] sin imagen para {ann}", file=sys.stderr)
    return pairs


def load_boxes(ann_path: Path):
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    boxes = []
    for o in data.get("objects", []):
        if o.get("geometryType") not in (None, "rectangle"):
            continue
        (x1, y1), (x2, y2) = o["points"]["exterior"][:2]
        boxes.append((o["classTitle"], min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def class_names(root: Path, pairs):
    meta = root / "meta.json"
    if meta.exists():
        names = [c["title"] for c in json.loads(meta.read_text(encoding="utf-8")).get("classes", [])]
        if names:
            return names
    seen = sorted({b[0] for _, a in pairs for b in load_boxes(a)})
    return seen


def starts(length, tile, step):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, step))
    s.append(length - tile)
    return s


def group_and_crop_of(img_name: str, ann_path: Path, regex):
    """Extrae el grupo (sitio/vuelo) y cultivo usando regex o tags del JSON de Supervisely."""
    if regex:
        m = re.search(regex, img_name)
        grp = m.group(1) if m and m.groups() else (m.group(0) if m else "otros")
        return grp, None

    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        tags = {t.get("name"): t.get("value") for t in data.get("tags", []) if isinstance(t, dict)}
        site = tags.get("site")
        flight = tags.get("flight_id")
        crop = tags.get("crop")
        if site is not None and flight is not None:
            return f"{site}_flight{flight}", crop
        elif site is not None:
            return str(site), crop
    except Exception:
        pass

    return img_name.split("_")[0], None


def split_groups(groups, group_crops, val, test, seed):
    """Asigna grupos enteros a train/val/test estratificando por cultivo si es posible."""
    if group_crops and any(group_crops.values()):
        crop_to_groups = defaultdict(list)
        for g in sorted(groups):
            crop_to_groups[group_crops.get(g, "otros")].append(g)

        res = {}
        rng = random.Random(seed)
        for c, grps in sorted(crop_to_groups.items()):
            g = list(grps)
            rng.shuffle(g)
            n = len(g)
            if n == 1:
                res[g[0]] = "train"
            elif n == 2:
                res[g[0]] = "val"
                res[g[1]] = "train"
            else:
                n_test = max(1, round(n * test))
                n_val = max(1, round(n * val))
                for x in g[:n_test]:
                    res[x] = "test"
                for x in g[n_test:n_test + n_val]:
                    res[x] = "val"
                for x in g[n_test + n_val:]:
                    res[x] = "train"
        return res

    g = sorted(groups)
    random.Random(seed).shuffle(g)
    n = len(g)
    if n < 3:
        return None
    n_test = max(1, round(n * test))
    n_val = max(1, round(n * val))
    return {**{x: "test" for x in g[:n_test]},
            **{x: "val" for x in g[n_test:n_test + n_val]},
            **{x: "train" for x in g[n_test + n_val:]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("salida")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--min-visible", type=float, default=0.5,
                    help="fraccion minima del area de la caja que debe quedar dentro del tile")
    ap.add_argument("--group-regex", default=None,
                    help="regex para extraer el grupo (sitio/vuelo) del nombre; default: tags Supervisely o texto antes del '_'")
    ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root, out = Path(a.dataset), Path(a.salida)
    pairs = find_pairs(root)
    if not pairs:
        sys.exit("No encontre pares img/ann. Revisa la ruta (debe contener <ds>/img y <ds>/ann).")
    names = class_names(root, pairs)
    cid = {n: i for i, n in enumerate(names)}

    img_groups = {}
    group_crops = {}
    for img, ann in pairs:
        grp, crop = group_and_crop_of(img.name, ann, a.group_regex)
        img_groups[img.name] = grp
        if crop:
            group_crops[grp] = crop

    count = defaultdict(int)
    for g in img_groups.values():
        count[g] += 1
    print(f"Grupos detectados ({len(count)}):", dict(count))
    gsplit = split_groups(set(img_groups.values()), group_crops, a.val, a.test, a.seed)
    if gsplit is None:
        print("[aviso] menos de 3 grupos: se divide por imagen (puede haber fuga de datos entre splits). "
              "Usa --group-regex para indicar como identificar sitio/vuelo.", file=sys.stderr)
        imgs = sorted(p[0].name for p in pairs)
        random.Random(a.seed).shuffle(imgs)
        n_t, n_v = max(1, round(len(imgs) * a.test)), max(1, round(len(imgs) * a.val))
        isplit = {**{n: "test" for n in imgs[:n_t]}, **{n: "val" for n in imgs[n_t:n_t + n_v]},
                  **{n: "train" for n in imgs[n_t + n_v:]}}
        get_split = lambda name: isplit[name]
    else:
        get_split = lambda name: gsplit[img_groups[name]]

    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    step = a.tile - a.overlap
    stats = defaultdict(lambda: [0, 0])  # split -> [tiles, cajas]
    for img_path, ann_path in pairs:
        split = get_split(img_path.name)
        boxes = load_boxes(ann_path)
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        for y0 in starts(H, a.tile, step):
            for x0 in starts(W, a.tile, step):
                x1, y1 = min(x0 + a.tile, W), min(y0 + a.tile, H)
                lines = []
                for cls, bx1, by1, bx2, by2 in boxes:
                    cx1, cy1, cx2, cy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    area, full = (cx2 - cx1) * (cy2 - cy1), max((bx2 - bx1) * (by2 - by1), 1e-9)
                    if area / full < a.min_visible:
                        continue
                    tw, th = x1 - x0, y1 - y0
                    lines.append(f"{cid[cls]} {((cx1 + cx2) / 2 - x0) / tw:.6f} {((cy1 + cy2) / 2 - y0) / th:.6f} "
                                 f"{(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}")
                stem = f"{img_path.stem}_x{x0}_y{y0}"
                im.crop((x0, y0, x1, y1)).save(out / "images" / split / f"{stem}.jpg", quality=95)
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                stats[split][0] += 1; stats[split][1] += len(lines)

    # path relativo para portabilidad local y en Google Colab
    yaml = ["path: .", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for n, i in cid.items()]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print("Clases:", cid)
    for s in ("train", "val", "test"):
        print(f"{s}: {stats[s][0]} tiles, {stats[s][1]} cajas")
    print(f"Listo -> {out/'data.yaml'}")


if __name__ == "__main__":
    main()
