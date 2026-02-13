import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# ==========================================
# 1. SETUP & KONFIGURASI
# ==========================================
# Ganti path ini sesuai lokasi dataset Anda
DATA_DIR = os.path.join('dataset', 'kentang') 
IMG_SIZE = 224
BATCH_SIZE = 32  # Jika Laptop "Out of Memory", turunkan jadi 16 atau 8
EPOCHS = 15      # ResNet butuh waktu belajar agak lama

# Cek GPU
print(f"TensorFlow Version: {tf.__version__}")
if tf.config.list_physical_devices('GPU'):
    print("✅ Menggunakan GPU untuk Training (NVIDIA)")
else:
    print("⚠️ Menggunakan CPU (Akan lambat untuk ResNet)")

# ==========================================
# 2. LOAD DATASET
# ==========================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"✅ Kelas ditemukan: {class_names}")

# Simpan Label agar urutan tidak tertukar
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

# Optimasi buffer data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. MEMBANGUN MODEL RESNET-50
# ==========================================
print("🏗️ Membangun arsitektur ResNet-50...")

# Augmentasi Data (Biar model tidak manja/overfitting)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

# Load Base Model ResNet50
# include_top=False artinya kita buang lapisan klasifikasi asli (1000 kelas)
base_model = applications.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# ResNet50 punya 170+ layer. Kita bekukan semua dulu.
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    # Preprocessing ResNet (Penting: ResNet suka input range -1 s/d 1)
    # Ini trik agar preprocessing masuk ke dalam model TFLite nanti
    layers.Rescaling(1./127.5, offset=-1), 
    
    data_augmentation,
    base_model,
    
    # Head baru untuk klasifikasi penyakit kita
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3), # Dropout diperbesar sedikit karena ResNet parameternya banyak
    layers.Dense(256, activation='relu'), # Tambahan layer dense agar makin pintar
    layers.Dense(len(class_names), activation='softmax')
])

# Compile Model Awal
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. TRAINING TAHAP 1 (Feature Extraction)
# ==========================================
print("🚀 Training Tahap 1 (Base Beku)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ==========================================
# 5. TRAINING TAHAP 2 (Fine Tuning)
# ==========================================
# Ini rahasia agar ResNet akurat: Buka gembok layer terakhir
print("🔓 Membuka gembok layer ResNet untuk Fine Tuning...")
base_model.trainable = True

# Kita hanya latih 30 layer terakhir ResNet (agar tidak merusak bobot imagenet)
# ResNet50 punya sekitar 175 layer
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Perlu compile ulang dengan learning rate SANGAT KECIL
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Training Tahap 2 (Fine Tuning)...")
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5) # Tambah 5 epoch lagi

# ==========================================
# 6. EXPORT KE TFLITE (DENGAN KOMPRESI)
# ==========================================
print("💾 Mengonversi ke TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- TEKNIK OPTIMASI (Wajib untuk ResNet) ---
# Ini akan mengubah bobot Float32 menjadi Int8 (Ukuran turun 4x lipat)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Simpan
filename = 'model_kentang_resnet.tflite'
with open(filename, 'wb') as f:
    f.write(tflite_model)

print("\n" + "="*50)
print(f"🎉 SUKSES! Model tersimpan: {filename}")
print("Jangan lupa ubah path di app.py Anda menjadi file model baru ini.")
print("="*50)

# Plot Grafik
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Mulai Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy (ResNet50)')
plt.show()