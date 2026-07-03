// Menu hamburger (mobile): toggle nav + tutup saat klik link / luar area.
(function () {
  const btn = document.getElementById("menuBtn");
  const nav = document.getElementById("nav");
  if (!btn || !nav) return;
  const setOpen = (open) => {
    nav.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  };
  btn.addEventListener("click", (e) => { e.stopPropagation(); setOpen(!nav.classList.contains("open")); });
  nav.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setOpen(false)));
  document.addEventListener("click", (e) => {
    if (nav.classList.contains("open") && !nav.contains(e.target) && e.target !== btn) setOpen(false);
  });
})();

// Demo: unggah/jatuhkan gambar -> POST byte mentah ke /api/predict -> tampilkan hasil.
(function () {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("file"), pick = $("pick");
  const dropContent = $("dropContent"), samplesEl = $("samples");
  const verdict = $("verdict"), conf = $("conf");
  const pNormal = $("pNormal"), pPothole = $("pPothole");
  const fNormal = $("fNormal"), fPothole = $("fPothole");

  const SAMPLES = [
    "samples/pothole_1.jpg", "samples/pothole_2.jpg", "samples/pothole_3.jpg",
    "samples/normal_1.jpg", "samples/normal_2.jpg", "samples/normal_3.jpg",
  ];
  SAMPLES.forEach((src) => {
    const img = new Image();
    img.src = src; img.alt = "contoh";
    img.addEventListener("click", () => classify(fetch(src).then((r) => r.blob())));
    samplesEl.appendChild(img);
  });

  function preview(blob) {
    const url = URL.createObjectURL(blob);
    dropContent.innerHTML = "";
    const img = new Image();
    img.src = url; img.alt = "pratinjau";
    img.onload = () => URL.revokeObjectURL(url);
    dropContent.appendChild(img);
  }

  function busy() {
    verdict.className = "verdict idle";
    verdict.textContent = "Memproses…";
    conf.textContent = "menjalankan ensembel LeNet-5 (NumPy)…";
  }

  function render(d) {
    if (d.error) {
      verdict.className = "verdict idle";
      verdict.textContent = "Gagal memproses";
      conf.textContent = d.error;
      return;
    }
    const pn = d.proba.normal, pp = d.proba.pothole;
    const isP = d.label === "pothole";
    verdict.className = "verdict " + d.label;
    verdict.textContent = isP ? "Terdeteksi berlubang." : "Jalan terlihat normal.";
    conf.textContent = "keyakinan " + (Math.max(pn, pp) * 100).toFixed(1) + "%";
    pNormal.textContent = (pn * 100).toFixed(1) + "%";
    pPothole.textContent = (pp * 100).toFixed(1) + "%";
    fNormal.style.width = (pn * 100).toFixed(1) + "%";
    fPothole.style.width = (pp * 100).toFixed(1) + "%";
  }

  async function classify(blobOrPromise) {
    busy();
    try {
      const blob = await blobOrPromise;
      preview(blob);
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
    if (e.target.files[0]) classify(Promise.resolve(e.target.files[0]));
  });
  if (pick) pick.addEventListener("click", (e) => { e.preventDefault(); fileInput.click(); });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) classify(Promise.resolve(f));
  });
})();
