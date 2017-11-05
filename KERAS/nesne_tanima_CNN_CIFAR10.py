# -*- coding: utf-8 -*- 
'''
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.

Amaç: Fotoğraftaki nesneyi sınıflandırmak.
Veriseti: CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)

Eğer arkaplanda TensorFlow kullanıyorsanız otomatik olarak GPU kullanılacaktır.
Theano arkaplanında GPU kullanarak çalıştırmak için gereken komut:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

25 epoch sonunda hata oranı 0.65'e, 50 epoch sonunda 0.55'e düşürüldü.
(Bu durumda hala yetersiz durumda.)
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os

batch_size = 32 # Her iterasyonda 32 fotoğraf alınır.
num_classes = 10 # CIFAR10 veri setinde 10 sınıf bulunmakta.
epochs = 200 # 200 epoch ile eğitim yapılacaktır.
data_augmentation = True # Canlı veri arttırması yapılacaktır.
save_dir = os.path.join(os.getcwd(), 'saved_models') # Modelin kaydedileceği yer belirlenir.
model_name = 'keras_cifar10_trained_model.h5' # Kaydedilecek modelin dosya adı belirlenir.

# Veri karıştırılır ve train-test şeklinde bölünür.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples') # Eğitim için örnek sayısı.
print(x_test.shape[0], 'test samples') # Test için örnek sayısı.

# Sınıflar ikili (binary) formununa dönüştürülür.
# "to_catogorical" fonksiyonu ile one-hot-encoding yapılmakta.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# 32 adet 3x3 boyutunda filtereler oluşturulur ve modele eklenir.
# "Padding" fotoğrafa çervçeve ekler ve çıkış boyutunun giriş boyutuna eşit olması sağlanır.
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
# ReLu aktivasyon fonksiyonumuzu ekliyoruz:
model.add(Activation('relu'))
# 32 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
model.add(Conv2D(32, (3, 3)))
# ReLu aktivasyon fonksiyonumuzu ekliyoruz:
model.add(Activation('relu'))
# 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı ekliyoruz:
model.add(MaxPooling2D(pool_size=(2, 2)))
# Rastgele olacak şekilde nöronların %25'ini kapatıyoruz: (Eğitim sırasındaki ezberlemeyi önlemek için.)
model.add(Dropout(0.25))

# 64 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
model.add(Conv2D(64, (3, 3), padding='same'))
# ReLu aktivasyon fonksiyonumuzu ekliyoruz:
model.add(Activation('relu'))
# 64 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
model.add(Conv2D(64, (3, 3)))
# ReLu aktivasyon fonksiyonumuzu ekliyoruz:
model.add(Activation('relu'))
# 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı ekliyoruz:
model.add(MaxPooling2D(pool_size=(2, 2)))
# Rastgele olacak şekilde nöronların %25'ini kapatıyoruz:
model.add(Dropout(0.25))

# 2 boyutlu görsellerimizi 1 boyutlu vektöre çeviriyoruz:
model.add(Flatten())
# 512 nöronumuzu modelimize ekliyoruz:
model.add(Dense(512))
# ReLu aktivasyon fonksiyonumuzu ekliyoruz:
model.add(Activation('relu'))
# Rastgele olacak şekilde nöronların %50'sini kapatıyoruz:
model.add(Dropout(0.5))
# 10 sınıfımızı temsil edecek 10 nöronumuzu modelimize ekliyoruz:
model.add(Dense(num_classes)) # num_classes = 10
# Sınıfların olasılıklarını hesaplamak için "Softmax" fonksiyonumuzu ekliyoruz:
model.add(Activation('softmax'))

# "RMSprop" optimizasyon fonksiyonumuzu hazırlıyoruz:
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Modeli eğitirken kullanacağımız optimizasyon ve hata hesaplama fonksiyonumuzu belirliyoruz:
# Sınıflandırma yapacağımız için "categorical_crossentropy" fonksiyonunu kullanıyoruz.
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Verimizi kesirli hale çeviriyoruz: (Örneğin: 255 -> 255.0)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Verimizi forma sokuyoruz. (Örneğin: 255.0 -> 1.0)
x_train /= 255
x_test /= 255

if not data_augmentation:
    # Veri arttırma işlemi yapılmayacak ise:
    print('Not using data augmentation.')
    # Modelimize verilerimizi veriyoruz ve eğitimimizi başlatıyoruz:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    # Veri arttırma işlemi yapılacak ise:
    print('Using real-time data augmentation.')
    # Canlı veri arttırmak için ayarlarımızı yapıyoruz:
    datagen = ImageDataGenerator(
                                 featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=0,
                                 width_shift_range=0.1,  # Görüntüleri rasgele olarak yatay olarak kaydırın.
                                 height_shift_range=0.1,  # Görüntüleri rasgele olarak dikey olarak kaydırın.
                                 horizontal_flip=True,  # Fotoğrafı yatay düzlemde rastgele çevirme.
                                 vertical_flip=False)

    # Veri oluşturma için hesaplama
    datagen.fit(x_train)

    # Canlı olarak arttırılan veri ile modelimizi eğitelim:
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Eğitilmiş modelimizi kaydedelim:
if not os.path.isdir(save_dir):
    # Klasör yoksa oluşturulur:
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Modelimizin başarısını ölçelim:
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
