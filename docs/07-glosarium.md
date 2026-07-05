# Kamus Istilah — Glosarium Ramah Awam

> 💡 **Untuk pembaca awam:** kamus ini menerjemahkan istilah teknis ke bahasa
> sehari-hari. Setiap entri berisi satu kalimat definisi ditambah satu analogi.
> Anda tidak perlu membaca berurutan — pakai saja sebagai rujukan saat menemui
> istilah asing di dokumen lain atau di makalah.

**Tensor** — wadah angka berdimensi banyak yang menyimpan gambar atau hasil hitungan; seperti tumpukan lembar spreadsheet: satu angka (nilai), satu baris (deret), satu tabel (lembar), lalu banyak tabel ditumpuk.

**Matriks** — tabel angka dua dimensi (baris × kolom); persis seperti papan catur yang tiap kotaknya diisi satu angka.

**Vektor** — deretan angka satu baris; ibarat daftar belanja bernomor, tiap posisi punya makna tetap.

**Batch / mini-batch** — sekelompok kecil gambar yang diproses bersamaan dalam satu langkah belajar; seperti guru memeriksa 32 lembar ujian sekaligus, bukan satu-satu atau seluruh kelas sekaligus.

**Epoch** — satu putaran penuh saat model sudah melihat semua gambar latih tepat satu kali; ibarat membaca ulang seluruh buku dari halaman pertama sampai terakhir sekali jalan.

**Reshape** — menyusun ulang angka yang sama ke bentuk berbeda tanpa mengubah isinya; seperti menata 12 telur dari satu baris panjang menjadi kotak 3×4 — telurnya tetap 12.

**Transpos** — menukar baris jadi kolom pada sebuah tabel; seperti memutar tabel jadwal 90 derajat sehingga judul yang tadinya di atas kini di samping.

**Broadcasting** — trik agar angka kecil bisa "diregangkan" otomatis untuk berpasangan dengan tabel besar; seperti satu resep bumbu yang diterapkan ke semua porsi masakan tanpa menulis ulang resepnya.

**Fancy indexing** — mengambil banyak elemen sekaligus dengan menyebut daftar posisinya; seperti berkata "ambilkan kursi nomor 3, 7, dan 10" alih-alih menunjuk satu per satu.

**In-place** — mengubah isi wadah angka langsung di tempat tanpa membuat salinan baru; seperti mengedit dokumen asli, bukan menyimpannya sebagai file baru dulu.

**Mask** — daftar benar/salah yang menandai elemen mana yang dipakai dan mana yang diabaikan; seperti stabilo yang menyorot kata-kata penting dan membiarkan sisanya.

**Argmax** — mencari posisi angka terbesar dalam sebuah deret; seperti menunjuk siswa dengan nilai tertinggi di daftar, bukan menyebut nilainya.

**Gradien** — penunjuk arah dan seberapa besar suatu nilai harus diubah agar hasil membaik; seperti kompas yang menunjukkan ke mana harus melangkah supaya lebih cepat sampai puncak.

**Backpropagation** — cara menelusuri kesalahan dari hasil akhir mundur ke setiap bagian model untuk tahu siapa yang perlu diperbaiki; seperti menelusuri balik resep gagal untuk menemukan bumbu mana yang salah takar.

**Forward & backward** — dua arah proses: maju (forward) menghasilkan tebakan, mundur (backward) menghitung kesalahan dan cara memperbaikinya; seperti mengerjakan soal lalu mencocokkan dengan kunci untuk tahu di mana keliru.

**Softmax** — mengubah sekumpulan skor mentah menjadi persentase keyakinan yang totalnya 100%; seperti mengubah perolehan suara jadi persentase pemilih tiap kandidat.

**Cross-entropy** — ukuran seberapa jauh tebakan model meleset dari jawaban benar; makin yakin tapi salah, makin besar hukumannya — seperti denda yang membengkak kalau kita menjawab "pasti benar" padahal salah.

**One-hot** — cara menulis jawaban kategori sebagai deret 0 dengan satu angka 1 di posisi yang benar; seperti kartu absensi yang hanya satu kotak dicentang untuk menandai kelas yang dihadiri.

**Momentum** — teknik agar perbaikan model tidak berhenti-henti melainkan mempertahankan arah seperti benda yang sudah melaju; seperti bola menggelinding menuruni bukit yang tetap meluncur melewati lubang-lubang kecil.

**Varians** — ukuran seberapa berpencar-pencar hasil dari nilai rata-ratanya; seperti tembakan panah yang tersebar jauh dari titik pusat menunjukkan varians tinggi.

**Overfitting** — kondisi saat model terlalu hafal contoh latihan sampai gagal pada soal baru; seperti murid yang menghafal kunci jawaban ujian lama tapi bingung menghadapi soal yang sedikit diubah.

**Regularisasi** — sekumpulan cara menahan model agar tidak terlalu hafal, mencakup *dropout* (mematikan sebagian bagian secara acak saat berlatih) dan *weight decay* (menghukum angka-angka yang membesar berlebihan); seperti melatih tim dengan pemain bergantian diistirahatkan dan melarang satu bintang mendominasi.

**Kernel / filter** — jendela kecil berisi angka yang digeser ke seluruh gambar untuk mendeteksi pola tertentu; seperti stempel bermotif yang dicap ke tiap bagian gambar untuk menemukan garis atau tepi.

**Stride** — seberapa jauh jendela filter melompat setiap kali digeser; seperti langkah kaki — langkah lebar (stride besar) memindai lebih cepat tapi lebih kasar.

**Padding** — pinggiran angka nol yang ditambahkan mengelilingi gambar agar tepinya tetap terbaca; seperti bingkai kosong di sekeliling foto supaya sudut-sudutnya tidak terpotong saat dicap filter.

**Feature map** — gambar hasil setelah sebuah filter dijalankan, menyorot di mana pola yang dicari muncul; seperti peta panas yang menyala di lokasi-lokasi yang cocok dengan motif stempel.

**Konvolusi** — proses menggeser filter ke seluruh gambar sambil menghitung kecocokan di tiap posisi; seperti menyorotkan senter bermotif ke seluruh dinding untuk melihat bagian mana yang berpola sama.

**Normalisasi / standardisasi** — menyetel ulang angka agar berada pada skala setara, biasanya berpusat di rata-rata 0 dengan sebaran seragam; seperti menyamakan satuan agar tinggi badan dan berat badan bisa dibandingkan adil.

**Ensembel** — menggabungkan tebakan beberapa model lalu mengambil kesepakatannya untuk hasil lebih stabil; seperti panel juri — rata-rata banyak juri lebih tepercaya daripada satu juri saja.

**TTA (*Test-Time Augmentation*)** — memberi model beberapa versi gambar yang sama (dibalik, digeser) saat pengujian lalu merata-ratakan hasilnya; seperti melihat objek dari beberapa sudut sebelum memutuskan itu apa.

**Notasi ilmiah** — cara ringkas menulis angka sangat besar atau sangat kecil, misalnya `1e-8` berarti 0,00000001; seperti singkatan "8 nol di belakang koma" supaya tak perlu menulis deretan nol yang panjang.

## Simbol matematika

**Σ (sigma)** — perintah "jumlahkan semuanya"; seperti tombol total di kalkulator yang menambahkan seluruh angka dalam daftar.

**∂L/∂x** — seberapa besar hasil akhir (L) berubah bila x digeser sedikit, sekaligus arah perbaikannya; seperti mengukur berapa banyak rasa sup berubah kalau garam ditambah sejumput.

**∇ dan δ (nabla & delta)** — gradien, yaitu sinyal koreksi yang mengalir mundur ke tiap bagian model; seperti umpan balik dari pelanggan yang diteruskan mundur ke setiap bagian produksi untuk diperbaiki.

**ᵀ (transpos)** — tanda kecil yang berarti "tukar baris jadi kolom" pada sebuah tabel; seperti memutar tabel 90 derajat.

**𝒩 (skrip N)** — angka acak yang diambil dari sebaran berbentuk lonceng, banyak di tengah dan jarang di tepi; seperti tinggi badan orang: kebanyakan sedang, sedikit yang sangat pendek atau sangat tinggi.

**e (bilangan Euler)** — bilangan tetap ≈2,718 yang jadi dasar hitungan pertumbuhan dan eksponen; seperti angka istimewa alam, mirip peran π pada lingkaran.

**μ (mu)** — rata-rata sekumpulan angka; seperti nilai tengah rapor yang mewakili keseluruhan.

**σ (sigma kecil)** — simpangan baku, ukuran seberapa lebar angka menyebar dari rata-ratanya; seperti mengukur apakah nilai satu kelas rapat mengumpul atau berpencar jauh.

**ε (epsilon)** — angka super kecil yang ditambahkan sebagai penjaga agar tak terjadi pembagian dengan nol; seperti ganjalan tipis supaya mesin hitung tidak macet.
