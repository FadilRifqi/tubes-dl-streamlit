# Presensi Wajah (Face Attendance) - Streamlit + PyTorch

Aplikasi presensi sederhana menggunakan Streamlit dan model PyTorch (facenet-pytorch) untuk deteksi wajah dan pengenalan berbasis embedding.

Fitur:
- Daftarkan pengguna baru (ambil foto atau unggah foto) dan simpan embedding wajah.
- Presensi: deteksi wajah dari kamera atau unggah foto, cocokkan dengan database embedding, dan catat kehadiran.

Persyaratan dan instalasi (Windows PowerShell):

1) (Direkomendasikan) Pasang PyTorch CPU first:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2) Pasang paket lain:

```powershell
pip install -r requirements.txt
```

Catatan: `facenet-pytorch` akan mengunduh model pralatih (MTCNN dan ResNet) pada penggunaan pertama.

Menjalankan aplikasi:

```powershell
streamlit run main.py
```

Folder data:
- `data/embeddings.npy` - embeddings yang tersimpan
- `data/names.pkl` - daftar nama sesuai urutan embeddings
- `data/attendance.csv` - catatan kehadiran (timestamp)
- `data/faces/` - foto wajah yang disimpan saat registrasi

Tips:
- Jika tidak memakai GPU, pastikan menginstal versi CPU dari PyTorch.
- Threshold kemiripan (similarity) default berada di 0.6 dan dapat diubah jika perlu.

Classifier (ViT) integration:
- Untuk memakai model classifier (ViT) yang sudah Anda latih, taruh file model di `model/model_vit_tuned_martua.pth` dan file label di `model/labels_pytorch.txt` (satu label per baris).
- Setelah meletakkan file tersebut, jalankan aplikasi; pada sidebar pilih metode pengenalan `Classifier (ViT)`.
- Jika model tidak terdeteksi atau gagal dimuat, aplikasi akan menampilkan peringatan.

Pastikan `torch` dan `torchvision` terpasang (lihat langkah instalasi di atas).

Lisensi: bebas untuk penggunaan pribadi/eksperimen.
