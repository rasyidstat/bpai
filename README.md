# bpjsai

## Data

* 57971 data training, 24633 data testing
* Terdapat 490 `kddati2` (kode wilayah) untuk training dan 489 untuk testing

## Petunjuk Penggunaan

Perangkat yang dibutuhkan untuk menjalankan kode ini adalah:

* Python 3.8.5
* Library
    * numpy 1.19.5
    * pandas 1.2.0
    * scikit-learn 0.24.1
    * lightgbm 3.1.1

Berikut ini tahapan yang perlu dilakukan untuk menghasilkan model dan prediksi:

1. Install Python sesuai dengan versi yang disebutkan di atas
2. Jalankan `pip install -r requirements.txt` untuk menginstall library yang dibutuhkan
3. Jalankan `python train_predict.py` untuk menghasilkan model dan prediksi

Di dalam direktori ini juga sudah disediakan file `Dockerfile` yang digunakan untuk menjalankan Docker sehingga kode dapat dijalankan di environment mana pun baik Windows, Mac maupun Linux dan hasilnya dipastikan sama persis. 

1. Install Docker
2. Jalankan `docker build -t bpjsai .` untuk menginstall semua perangkat yang dibutuhkan menggunakan Docker
3. Jalankan `docker run -it bpjsai` untuk menjalankan Docker

## Ceklis Pengembangan Model

- [x] Percobaan model pertama menggunakan LightGBM untuk prediksi kasus dan biaya 
- [ ] Menggunakan Docker dan membuat dokumentasi ringkas tentang penggunaan kode
- [ ] Percobaan model kedua dengan optimasi MAPE untuk prediksi kasus
- [ ] Kapan pencilan muncul? Apakah secara acak atau tidak?