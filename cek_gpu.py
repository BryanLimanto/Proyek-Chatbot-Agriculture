import tensorflow as tf

# Cek apakah GPU tersedia
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Cek nama GPU yang terdeteksi
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Name:", tf.config.list_physical_devices('GPU')[0])
else:
    print("GPU tidak terdeteksi, sistem akan menggunakan CPU.")