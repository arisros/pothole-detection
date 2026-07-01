# Image situs + demo model deteksi jalan berlubang (LeNet-5 NumPy dari nol).
# Satu proses Python (stdlib + numpy + pillow) menyajikan statis & /api/predict.
FROM python:3.11-slim

# Hanya numpy & pillow (inti CNN murni NumPy, bukan framework DL).
RUN pip install --no-cache-dir "numpy>=1.24" "pillow>=9.0"

WORKDIR /app
# app/ : server.py, cnn/, weights.npz, norm_stats.json
COPY app/ /app/
# www/ : situs statis (index, viewer, css, js, figures, content, samples)
COPY www/ /app/www/

ENV POTHOLES_HOME=/app \
    POTHOLES_STATIC=/app/www \
    POTHOLES_PORT=80 \
    POTHOLES_BIND=0.0.0.0

EXPOSE 80
CMD ["python", "server.py"]
