# Makefile pipeline deteksi jalan berlubang (CNN dari nol)
PY ?= python3

.PHONY: all data label preprocess gradcheck train eval visualize clean

all: data label preprocess gradcheck train eval visualize

data:           ## Unduh dataset publik pothole/normal
	$(PY) src/download_data.py

label:          ## Susun kelas + labels.csv
	$(PY) src/label_dataset.py

preprocess:     ## Grayscale, resize, normalisasi, split -> dataset.npz
	$(PY) src/preprocess.py

gradcheck:      ## Numerical gradient checking (bukti backprop benar)
	$(PY) -m src.cnn.gradcheck

train:          ## Latih LeNet-5 manual -> weights.npz + history.csv
	$(PY) src/train.py

eval:           ## Evaluasi pada test set + figures
	$(PY) src/evaluate.py

visualize:      ## Render feature map antar lapisan
	$(PY) src/visualize.py

clean:          ## Hapus artefak hasil (bukan data mentah)
	rm -f $(wildcard experiments/weights.npz experiments/history.csv)
	rm -f $(wildcard experiments/figures/*.png)
	rm -f $(wildcard data/processed/*.npz data/processed/*.json)
