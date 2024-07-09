import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Fungsi untuk membaca gambar
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"File tidak ditemukan di path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke format RGB
    return image

# Fungsi untuk inisialisasi titik data secara acak ke salah satu k cluster
def initialize_clusters(data, k):
    n_samples = data.shape[0]
    labels = np.random.randint(0, k, n_samples)
    return labels

# Fungsi untuk menghitung pusat cluster
def calculate_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        centroids[i] = points.mean(axis=0) if len(points) > 0 else np.random.rand(data.shape[1])
    return centroids

# Fungsi untuk segmentasi menggunakan K-Means dengan inisialisasi manual
def segment_image(image, k=3, max_iter=100):
    # Mengubah gambar menjadi array 2D di mana setiap baris adalah sebuah piksel
    data = image.reshape((-1, 3)).astype(np.float32)

    # Inisialisasi acak titik data ke cluster
    labels = initialize_clusters(data, k)
    
    # Ulangi hingga konvergen atau mencapai jumlah iterasi maksimal
    for iteration in range(max_iter):
        centroids = calculate_centroids(data, labels, k)
        new_labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    
    # Konversi pusat cluster ke tipe uint8
    centroids = np.uint8(centroids)
    
    # Tetapkan label cluster ke piksel gambar
    segmented_image = centroids[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

# Fungsi untuk menyimpan gambar hasil segmentasi
def save_image(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, image):
        raise IOError(f"Tidak dapat menyimpan gambar ke path: {path}")

# Main program
if __name__ == "__main__":
    input_path = "C:/PengolahanCitra_UAS/UASCITRA/image.jpg"  # Ganti dengan path gambar input Anda
    output_directory = "C:/PengolahanCitra_UAS/UASCITRA/Output/"  # Ganti dengan direktori untuk menyimpan gambar hasil
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_path = os.path.join(output_directory, "output_image2.jpeg")  # Nama file output

    try:
        image = load_image(input_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    k = 7  # Jumlah cluster
    segmented_image = segment_image(image, k)
    
    try:
        save_image(output_path, segmented_image)
    except IOError as e:
        print(e)
        exit()
    
    # Menampilkan gambar asli dan hasil segmentasi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image)
    plt.show()
    print(f"Hasil segmentasi telah disimpan di: {output_path}")
