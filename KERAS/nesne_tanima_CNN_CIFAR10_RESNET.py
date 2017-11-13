
"""
Amaç: Verilen görseldeki nesmeyi tanımak
Yöntem: CIFAR10 veri seti üzerinde ResNet eğitmek
Veriseti: CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)

50 epoch sonunda 91% üzerinde test doğruluk oranı elde ediliyor.
GTX 1080Ti kullanarak her bir epoch 48 saniye sürüyor.
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

# Parametrelerin eğitilmesi
batch_size = 32
epochs = 100
data_augmentation = True

# Ağ mimarisi parametreleri
num_classes = 10
num_filters = 64
num_blocks = 4
num_sub_blocks = 2
use_max_pool = False

# Veri setinin yüklenip eğitim/test olarak ayrılması
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Giriş resimlerinin boyutları
# Varsayılan veri formatı "channels_last".
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

if K.image_data_format() == 'channels_first':
    img_rows = x_train.shape[2]
    img_cols = x_train.shape[3]
    channels = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

# Verinin kesirli hale getirilip normalleştirilmesi
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Sınıf vektörlerinin ikili sınıf matrislerine dönüştürülmesi
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Start model definition.
inputs = Input(shape=input_shape)
x = Conv2D(num_filters,
           kernel_size=7,
           padding='same',
           strides=2,
           kernel_initializer='he_normal',
           kernel_regularizer=l2(1e-4))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Orjinal makale 1. evrişimden sonra max pool metodunu kullanmaktadır.
# use_max_pool = True olduğunda doğruluk oranı 87%'ye kadar çıkıyor.
# Cifar10 veri setinin resimleri max pool metodunu kullanmak için çok küçük(32x32). Bu nedenle bu yöntemi atlıyoruz.
if use_max_pool:
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    num_blocks = 3

# Modelin temelinin oluşturulması
for i in range(num_blocks):
    for j in range(num_sub_blocks):
        strides = 1
        is_first_layer_but_not_first_block = j == 0 and i > 0
        if is_first_layer_but_not_first_block:
            strides = 2
        y = Conv2D(num_filters,
                   kernel_size=3,
                   padding='same',
                   strides=strides,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(num_filters,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(y)
        y = BatchNormalization()(y)
        if is_first_layer_but_not_first_block:
            x = Conv2D(num_filters,
                       kernel_size=1,
                       padding='same',
                       strides=2,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

    num_filters = 2 * num_filters

# Sınıflandırıcının en başa eklenmesi
x = AveragePooling2D()(x)
y = Flatten()(x)
outputs = Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)

# Modelin oluşturulup derlenmesi
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

# Model dizinin hazırlanıp, kaydedilmesi
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_resnet_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate decaying.
# Modelin kaydedilmesi ve öğrenme değerinin azaltılması için geriçağırımların hazırlanması
checkpoint = ModelCheckpoint(filepath=filepath,
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]

# Eğitimin başlatılması. Veri arttırma ile ya da veri arttırma olmaksızın.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # Ön işleme ve gerçek zamanlı veri arttırımının uygulanması
    datagen = ImageDataGenerator(
        featurewise_center=False,  # Giriş verilerinin prtalamasının 0'lanması
        samplewise_center=False,  # Herbir örnek verinin ortalamasının 0'a eşitlenmesi
        featurewise_std_normalization=False, # Giriş verilerinin, veri setinin standart varyans değerine bölünmesi
        samplewise_std_normalization=False,  # Herbir verinin standart varyans değerine bölünmesi
        zca_whitening=False,  # "ZCA whitening" metodunun uygulanması
        rotation_range=0,  # Resimlerin bir sınır aralığında gelişi güzel döndürülmesi (degrees, 0 to 180)
        width_shift_range=0.1,  # Resimlerin gelişigüzel bir şekilde yatay olarak kaydırılması (toplam genişliğin bölümü)
        height_shift_range=0.1,  # Resimlerin gelişigüzel bir şekilde dikey olarak kaydırılması(toplam yüksekliğin bölümü)
        horizontal_flip=True,  # Resimlerin gelişigüzel bir şekilde yatay olarak çevirilmesi
        vertical_flip=False)  # Resimlerin gelişigüzel bir şekilde dikey olarak çevirilmesi

    # Normalizasyon gerekliliklerinin hesaplanması
    # (standart varyans, ortalama, ve asıl bileşenler eğer "ZCA whitening" uygulanıyorsa).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Eğitilmiş modelin başarısının ölçülmesi
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])