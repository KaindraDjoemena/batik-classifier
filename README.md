# Batik Classifier

Klasifikasi motif batik menggunakan PyTorch dengan arsitektur efficientnet_b0

## Dataset
- https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia
- https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs

### Pembuatan dataset
- menggabungkan dataset
- menghapus sample-sample yang kurang bagus (banyak noise ataupun beda gambar)
- partisi 0.6/0.2/0.2 setiap folder motif batik untuk membuat folder: training/, testing/, dan validation/

## Contoh app
![screenshot aplikasi](img/app.png)

## Akurasi model
![akurasi model dalam mendeteksi tipe batik](img/accuracy.png)
