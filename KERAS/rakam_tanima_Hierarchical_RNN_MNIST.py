# -*- coding: utf-8 -*- 

"""
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.
Amaç: El yazısı rakamların tanınması.
Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
Algoritma: Hiyerarşik Devirli Sinir Ağları (Hierarchical Recurrent Neural Networks (HRNN))

El yazısı rakamların tanınmasında Hiyerarşik Devirli Sinir Ağları (HRNN) kullanımının bir örneğidir.
HRNN'ler karmaşık bir dizilim üzerinde geçici hiyerarşilerin birçok katmanını kullanarak öğrenebilirler.
Genelde, HRNN'in ilk devirli katmanı cümleleri (kelime vektörü gibi) cümle vektörlerine dönüştürüyor.
İkinci devirli katmanı sonrasında bu vektörlerin dizilimini bir döküman vektörüne dönüştürüyor.
Bu döküman vektörü ile hem kelime seviyesindeki hem de cümle seviyesindeki içerik yapısının korunmuş oluyor.

# Referanslar
- [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
  Paragraf ve dökümanları HRNN ile dönüştürüyor.
  Sonuçların gösterdiğine göre HRNN standart RNN'lerin performansını geçmiş
  ve yazı özetlemek yada soru-cevap üretimi gibi daha ileri seviye görevlerde kullanılabilir.
- [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
  İskelet bazında hareket tahmini konusunda şimdiye kadarki en iyi sonuçlara ulaşmıştır.
  3 katmanlı çift yönlü (bidirectional) HRNN ve tamamen bağlı katmanların (fully connected layers) kombinasyonu kullanılmıştır.

Aşağıdaki MNIST örneğinde, birinci LSTM katmanı öncelikle herbir (28,1) boyutundaki piksel sütununu (128,) boyutunda sütun vektörüne dönüştürüyor.
İkinci LSTM sonrasında bu (28, 128) boyutundaki 28 sütun vektörünü resim vektörüne yani tüm resme dönüştürüyor.
Son yoğun (Dense) katman tahmin yapmak için eklendi.

5 epoch sonunda, eğitim verisetinde 98.58%, validasyon verisetinde 98.64% doğruluk oranı elde ediliyor.
"""

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

# Parametrelerin değerlerinin belirlenmesi.
batch_size = 32 # her bir iterasyonda "32" resim alınsın
num_classes = 10  # ayırt etmek istediğimiz "10" rakam
epochs = 5 # eğitim 5 epoch sürsün


# Modele yerleştirme boyutları (Embedding Dimensions).
row_hidden = 128
col_hidden = 128

# Veriseti yükleniyor ve eğitim/test diye ayrılıyor.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veri HRNN için 4 boyutlu olarak yeniden şekillendiriliyor.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Sınıf vektörleri ikili (binary) forma dönüştürülüyor.
# "to_catogorical" fonksiyonu ile one-hot-encoding yapıyoruz.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row, col, pixel = x_train.shape[1:]

# 4 boyutlu giriş verileri.
x = Input(shape=(row, col, pixel))

# Her satırın pikselleri TimeDistributed Wrapper kullanılar dönüştürülüyor (encoding).
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# Dönüştürülmüş satırların sütunları da dönüştürülüyor.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Son olarak, tahmin yapılıyor.
prediction = Dense(num_classes, activation='softmax')(encoded_columns)

# Model kuruluyor.
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Eğitim yapılıyor.
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Sonuçların değerlendirilmesi.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])