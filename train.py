import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- KONFIGURASI ---
DATA_DIR = "./dataset/kentang" # Sesuaikan dengan folder Anda
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 1. Load Dataset dari Direktori Lokal
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names

# 2. Membangun Model (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Proses Training
print("🚀 Memulai Training Lokal...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 4. Simpan Model & Label ke Folder 'model'
if not os.path.exists('model'): os.makedirs('model')

model.save("./model/plant_disease_model.keras")
with open("./model/labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ Selesai! Model disimpan di folder /model")