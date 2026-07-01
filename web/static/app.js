// Logika demo: unggah/jatuhkan gambar -> POST byte mentah ke /api/predict -> tampilkan hasil.
(function () {
  const drop = document.getElementById("drop");
  const dropCard = document.getElementById("dropcard");
  const pick = document.getElementById("pick");
  const fileInput = document.getElementById("file");
  const dropContent = document.getElementById("dropContent");
  const verdict = document.getElementById("verdict");
  const pNormal = document.getElementById("pNormal");
  const pPothole = document.getElementById("pPothole");
  const fNormal = document.getElementById("fNormal");
  const fPothole = document.getElementById("fPothole");
  const samplesEl = document.getElementById("samples");

  const SAMPLES = [
    "samples/pothole_1.jpg", "samples/pothole_2.jpg", "samples/pothole_3.jpg",
    "samples/normal_1.jpg", "samples/normal_2.jpg", "samples/normal_3.jpg",
  ];

  SAMPLES.forEach((src) => {
    const img = document.createElement("img");
    img.src = src;
    img.alt = "contoh";
    img.addEventListener("click", () => classifyUrl(src));
    samplesEl.appendChild(img);
  });

  function showPreview(srcOrBlob) {
    const url = typeof srcOrBlob === "string" ? srcOrBlob : URL.createObjectURL(srcOrBlob);
    const img = new Image();
    // once the preview image has real dimensions, ask the wired-card to redraw
    // its sketchy border so it fits the new content.
    img.onload = () => { if (dropCard && dropCard.requestUpdate) dropCard.requestUpdate(); };
    img.src = url;
    img.alt = "pratinjau";
    dropContent.innerHTML = "";
    dropContent.appendChild(img);
  }

  function setBusy() {
    verdict.className = "verdict idle";
    verdict.textContent = "Memproses…";
  }

  function render(data) {
    if (data.error) {
      verdict.className = "verdict idle";
      verdict.textContent = "Gagal: " + data.error;
      return;
    }
    const pn = data.proba.normal, pp = data.proba.pothole;
    verdict.className = "verdict " + data.label;
    verdict.textContent = data.label === "pothole"
      ? "🟠 Terdeteksi BERLUBANG" : "🟢 Jalan NORMAL";
    pNormal.textContent = (pn * 100).toFixed(1) + "%";
    pPothole.textContent = (pp * 100).toFixed(1) + "%";
    // wired-progress is driven by its `.value` property (0–100), not CSS width.
    fNormal.value = +(pn * 100).toFixed(1);
    fPothole.value = +(pp * 100).toFixed(1);
  }

  async function classifyBlob(blob) {
    setBusy();
    showPreview(blob);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": blob.type || "image/jpeg" },
        body: blob,
      });
      render(await res.json());
    } catch (e) {
      render({ error: String(e) });
    }
  }

  async function classifyUrl(url) {
    setBusy();
    showPreview(url);
    try {
      const blob = await (await fetch(url)).blob();
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": blob.type || "image/jpeg" },
        body: blob,
      });
      render(await res.json());
    } catch (e) {
      render({ error: String(e) });
    }
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) classifyBlob(e.target.files[0]);
  });
  if (pick) pick.addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) classifyBlob(f);
  });
})();
