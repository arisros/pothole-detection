# Changelog

Semua perubahan penting proyek ini. Format mengikuti gaya
[Keep a Changelog](https://keepachangelog.com/); versi merujuk **tag image rilis**
(`potholes:vN`) yang di-deploy ke potholes.arisjirat.com. Milestone model dicatat
terpisah di bawah.

## [v10], 2026-07-04, Lapisan ramah awam
### Added
- Kamus Istilah: glosarium `docs/07-glosarium.md` + LAMPIRAN B di makalah
  (~35 istilah & simbol matematika, tiap entri definisi + analogi).
- Halaman `docs/08-catatan-pengembangan.md` (perjalanan proyek) + `CHANGELOG.md`.
- 24 callout "Untuk awam" (analogi) setelah tiap rumus utama & pembuka bagian
  teknis LAMPIRAN A; kotak "cara membaca rumus" di §3.6; primer di kepala LAMPIRAN A.
- Gaya callout di `viewer.html`/`style.css` (`.callout-tip` / `.callout-math` ).
### Changed
- Halaman muka dijinakkan untuk awam: legend Conv/Pool/FC/Softmax, gloss "ensembel =
  panel juri", tooltip galat gradien, tautan Kamus Istilah. `style.css?v=7`.

## [v9], 2026-07-04, LAMPIRAN A: bedah kode
### Added
- LAMPIRAN A: bedah kode 17 modul (A.1-A.17: im2col/col2im, Conv fwd/bwd, ReLU,
  MaxPool, Flatten, Dense, Dropout, Softmax+CE, init, LeNet-5, SGD, Adam, trainer,
  augment, preprocess, ensembel+TTA), kode setia + penjelasan baris + diagram.

## [v8], 2026-07-04, Keterbacaan
### Added
- Syntax highlighting (highlight.js) untuk kode di makalah/docs.
### Changed
- Semua flowchart Mermaid dibuat vertikal (TB) agar pas di kolom diagram.

## [v7], 2026-07-04, Komponen kode↔diagram
### Added
- Komponen "codepair" (`:::pair`): kode & diagram side-by-side (desktop) / tab (HP).
### Changed
- Diagram diperbesar (SVG mengisi lebar kolom; font 13→15px).
### Fixed
- Diagram sequence training-loop error (simbol `;`/`√`/diakritik pada Note Adam).

## [v6], 2026-07-04, Model 94,4% live
### Added
- `web/server.py`: inferensi ensembel RGB 48×48 + test-time augmentation; bobot
  per-seed di-bundle via `build-image.sh`. Bobot ensembel di-commit ke repo.
### Changed
- Seluruh prosa (makalah, docs, `index.html`, PANDUAN, README) diselaraskan dari
  75,7% ke **94,4%** (RGB 48×48, regularisasi, Adam+cosine, ensembel+TTA).

---

## Milestone model

### Akurasi 75,70% → 94,39%, 2026-07-04 (commit `ed3039d`)
Rangkaian perbaikan (semua murni-NumPy):
- **RGB 48×48** dari grayscale 32×32 (pengungkit terbesar: ~79% → ~92,5%).
- **Regularisasi**: Dropout 0.3, L2 weight decay 1e-4, augmentasi daring per-batch.
- **Optimizer Adam** + jadwal cosine; 40 epoch; restore bobot val-terbaik.
- **Ensembel 3-seed** [42,7,123] + **TTA** (flip): ~92,5% → 94,39%.

Hasil test (107 citra): akurasi **94,39%** · presisi 91,53% · recall **98,18%** ·
F1 **94,74%** · confusion matrix TP 54 / TN 47 / FP 5 / FN 1.

### Baseline 75,70%, model awal
Grayscale 32×32, model tunggal, SGD+momentum, 15 epoch. Overfitting (train ~100%,
val ~80-84%). Presisi 78,43% · recall 72,73% · F1 75,47%.
