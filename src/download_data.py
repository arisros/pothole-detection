"""Unduh dataset citra jalan berlubang (pothole) vs jalan normal dari sumber publik.

Sumber utama: repositori GitHub publik (berlisensi MIT) yang menyimpan dataset
klasifikasi biner dengan struktur folder kelas:

    My Dataset/train/Pothole , My Dataset/test/Pothole   -> kelas "pothole"
    My Dataset/train/Plain   , My Dataset/test/Plain     -> kelas "normal"

Hanya berkas gambar yang diperlukan yang diunduh (lewat raw.githubusercontent.com),
bukan seluruh repositori, agar hemat bandwidth. Hasil disusun ke:

    data/raw/pothole/*.jpg
    data/raw/normal/*.jpg

Catatan konteks: dataset ini bersifat umum (bukan khusus Indonesia). Untuk data
khusus Indonesia (mis. Roboflow "Road Damage Indonesia" atau subset RDD2022
Indonesia) diperlukan API key; pipeline ini dirancang agar mudah diganti
sumbernya — cukup letakkan gambar pada data/raw/pothole dan data/raw/normal.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Sumber utama dan strategi pemetaan folder -> kelas.
PRIMARY_REPO = "anantSinghCross/pothole-detection-system-using-convolution-neural-networks"
PRIMARY_BRANCH = "master"
PRIMARY_ROOT = "My Dataset"
# Substring nama folder -> nama kelas keluaran.
FOLDER_TO_CLASS = {"Pothole": "pothole", "Plain": "normal"}

IMG_EXTS = (".jpg", ".jpeg", ".png")
TREE_API = "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def _http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "potholes-dl"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _list_image_paths(repo, branch, root):
    """Ambil daftar path gambar di bawah `root` beserta kelasnya."""
    url = TREE_API.format(repo=repo, branch=branch)
    tree = json.loads(_http_get(url))
    if "tree" not in tree:
        raise RuntimeError(f"Gagal membaca tree repo: {tree.get('message')}")

    items = []
    for node in tree["tree"]:
        path = node["path"]
        if not path.startswith(root + "/"):
            continue
        if not path.lower().endswith(IMG_EXTS):
            continue
        cls = None
        for folder, name in FOLDER_TO_CLASS.items():
            if f"/{folder}/" in path or path.split("/")[-2] == folder:
                cls = name
                break
        if cls is not None:
            items.append((path, cls))
    return items


def _download_one(repo, branch, path, dest):
    try:
        data = _http_get(RAW_URL.format(repo=repo, branch=branch, path=urllib.parse.quote(path)))
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! gagal {path}: {exc}")
        return False


def download_from_repo(repo, branch, root):
    print(f"Membaca daftar berkas dari {repo}@{branch} ...")
    items = _list_image_paths(repo, branch, root)
    if not items:
        raise RuntimeError("Tidak ada gambar ditemukan pada sumber.")

    # Siapkan folder kelas.
    for cls in set(c for _, c in items):
        os.makedirs(os.path.join(config.RAW_DIR, cls), exist_ok=True)

    print(f"Mengunduh {len(items)} gambar ...")
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for idx, (path, cls) in enumerate(items):
            ext = os.path.splitext(path)[1].lower()
            dest = os.path.join(config.RAW_DIR, cls, f"{cls}_{idx:05d}{ext}")
            tasks.append(ex.submit(_download_one, repo, branch, path, dest))
        ok = sum(1 for t in tasks if t.result())
    print(f"Selesai: {ok}/{len(items)} berhasil diunduh.")
    return items, ok


def write_source_note(repo, branch, root, items):
    counts = {}
    for _, cls in items:
        counts[cls] = counts.get(cls, 0) + 1
    note = (
        "Sumber Dataset\n"
        "==============\n"
        f"Repositori : https://github.com/{repo}\n"
        f"Branch     : {branch}\n"
        f"Folder     : {root}\n"
        "Lisensi    : MIT\n\n"
        "Pemetaan kelas:\n"
        "  Pothole -> pothole (jalan berlubang, label 1)\n"
        "  Plain   -> normal  (jalan normal,   label 0)\n\n"
        "Jumlah gambar per kelas:\n"
    )
    for cls, n in sorted(counts.items()):
        note += f"  {cls:8s}: {n}\n"
    os.makedirs(config.RAW_DIR, exist_ok=True)
    with open(os.path.join(config.RAW_DIR, "SOURCE.txt"), "w") as f:
        f.write(note)


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    try:
        items, ok = download_from_repo(PRIMARY_REPO, PRIMARY_BRANCH, PRIMARY_ROOT)
    except Exception as exc:  # noqa: BLE001
        print("GAGAL mengunduh dari sumber utama:", exc)
        print("Silakan letakkan gambar secara manual pada:")
        print(f"  {os.path.join(config.RAW_DIR, 'pothole')}/")
        print(f"  {os.path.join(config.RAW_DIR, 'normal')}/")
        return 1

    write_source_note(PRIMARY_REPO, PRIMARY_BRANCH, PRIMARY_ROOT, items)
    print(f"Dataset tersimpan di {config.RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
