#CNN - CIFAR-10
## Deskripsi

Project ini merupakan implementasi **Convolutional Neural Network (CNN)** untuk melakukan klasifikasi gambar pada dataset **CIFAR-10** menggunakan **TensorFlow/Keras**.

Model menerima citra berukuran **32 × 32 piksel dengan 3 channel warna (RGB)** dan mengklasifikasikan gambar ke dalam 10 kategori, yaitu:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Project dikerjakan menggunakan Python dan notebook Google Colab dengan dukungan GPU.

## Tujuan

Tujuan project ini adalah membangun dan mengevaluasi model CNN yang mampu mengenali objek pada gambar CIFAR-10 melalui tahapan:

1. Memuat dataset CIFAR-10.
2. Melakukan preprocessing data.
3. Melakukan data augmentation.
4. Membangun arsitektur CNN.
5. Melatih model.
6. Mengevaluasi performa model.
7. Menganalisis hasil menggunakan grafik training dan confusion matrix.
8. Melakukan prediksi terhadap gambar baru.

## Teknologi yang Digunakan

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Google Colab**
- **GPU NVIDIA T4** untuk proses komputasi

## Dataset

Project menggunakan dataset **CIFAR-10** yang tersedia melalui `tensorflow.keras.datasets.cifar10`.

Dataset memiliki 10 kelas gambar seperti yang telah dijelaskan pada bagian kategori. Data training dan testing dimuat langsung menggunakan fungsi `cifar10.load_data()`.

## Preprocessing

Sebelum digunakan untuk training, gambar dinormalisasi dari rentang nilai piksel 0–255 menjadi 0–1 dengan membagi nilai piksel menggunakan 255.

Label kelas kemudian diubah menjadi bentuk **one-hot encoding** menggunakan `to_categorical()` dengan jumlah 10 kelas.

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
