# -*- coding: utf-8 -*- 

'''
MNIST dataseti üzerinde Denoising Autoencoder (parazit arındıran otomatik kodlayıcı) uyguluyor.
Parazit arındırma otomatik kodlayıcıların en klasik uygulama alanlarından birisidir.
Parazit arındırma süreci asıl sinyalleri bozan istenmeyen parazitlerden kurtulmayı sağlar.

Parazit + Veri ---> Denoising Autoencoder ---> Veri
Bozuk bir veri kümesini girdi, asıl veriyi çıktı olarak verdiğimizde Denoising Autoencoder
asıl veriyi elde etmek için gizli yapıyı kurtarır.

Bu örnek modüler dizayna sahip. Encoder (şifreleyici), Decoder (çözümleyici) ve Autoencoder
aynı ağırlık değerlerini (weight) paylaşan 3 ayrı modeldir. Örneğin, autoencoder eğitildikten
sonra, encoder girdi verisetinin örtülü vektörlerini (latent vectors) oluşturmak için 
kullanılabilir. Böylece PCA ve TSNE'nin yaptığı gibi küçük boyutta indirgeyerek görselleştirme 
yapılabilir.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# Veriseti yükleniyor ve eğitim/test diye ayrılıyor.
(x_train, _), (x_test, _) = mnist.load_data()

# Veri 4 boyutlu olarak yeniden şekillendiriliyor.
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# MNIST verisine parazit katarak bozuyoruz. Bunun için
# 0.5 merkezinde std=0.5 olan bir normal frekans dağılımı ekliyoruz veriye.
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

# Parazit katılmış veriler 0 ile 1 arasını aşmayacak şekilde düzenleniyor.
# 0'dan küçük değerler 0'a, 1'den büyük değerler 1'e eşitleniyor.
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Model parametreleri
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16

# Encoder ve Decoder için CNN katmanları ve her katman için uygulanacak filtreler
# Burada 2 katman var. İlk katmanın filtre sayısı 32, ikincisinin 64.
layer_filters = [32, 64]


# Autoencoder Modelinin Kurulumu

# Öncelikle Encoder modeli kuruluyor
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

# Con2D blokları
# Not:
# 1) Derin ağlarda ReLU kullanmadan önce Batch Normalization kullanın
# 2) strides>1'a alternatif olarak MaxPooling2D kullanın
# - daha hızlı ama strides>1 kadar iyi değil
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Decoder Modelini oluştururken lazım olan boyut bilgileri
shape = K.int_shape(x)

# Örtülü vektörün (Latent Vector) oluşturulması
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Encoder Modelinin örneklendirilmesi
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Decoder Modelinin oluşturulması
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Transpozu alınmış Conv2D blokları
# Not:
# 1) Derin ağlarda ReLU kullanmadan önce Batch Normalization kullanın
# 2) strides>1'a alternatif olarak UpSampling2D kullanın
# - daha hızlı ama strides>1 kadar iyi değil
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Decoder Modelinin örneklendirilmesi
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Autoencoder Modelinin örneklendirilmesi
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Autoencoder'ı eğitiyoruz bu adımda
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

# Parazit eklenmiş test görsellerinin çıktılarını autoencoder ile tahmin ediyoruz
x_decoded = autoencoder.predict(x_test_noisy)

# İlk 8 parazit eklenmiş ve bozulmuş imajı görselliyoruz
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()