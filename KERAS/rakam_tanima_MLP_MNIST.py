# -*- coding: utf-8 -*- 
"""
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.

Amaç: El yazısı rakamların tanınması.
Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)

Microsoft Azure Notebook: 

20 epoch sonunda %98.40 test doğruluk oranı elde ediliyor.
"""
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128 # her bir iterasyonda "128" resim alınsın
num_classes = 10 # tanınmak istenen 0-9 rakam (10 sınıf)
epochs = 20 # eğitim 20 epoch(eğitim devir sayısı) sürsün

# the data, shuffled and split between train and test sets
# mnist veriseti rastgele karıştırılmış şekilde train ve set setleri olarak yükleniyor
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Sinir ağımız her eğitim örneği için tek bir vektör alacaktır, 
# bu nedenle girdiyi 28x28 resim tek bir 784 boyutlu vektör olarak şekilde yeniden şekillendiriyoruz.
# Ayrıca girdileri [0-255] yerine [0-1] aralığında ölçeklendireceğiz.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples(eğitim örnek sayısı)')
print(x_test.shape[0], 'test samples(test örnek sayısı)')

# sınıf vektörlerini ikili sınıf matrislerine dönüştürüyoruz
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Ağımızı kuralım
# Burada basit bir 3 katmanlı tam bağlantılı ağ yapacağız.(fully connected network ("Dense"))
# Ayrıca eğitim sırasında aşırı öğrenme (overfitting) olmaması için bırakma/atılma ("Dropout") uygulayacağız.
# Dropout tekniği 2014 yılında bir makale de önerilmiştir ve o zamandan beri benimsenerek kullanılmıştır. (http://jmlr.org/papers/v15/srivastava14a.html)
# Pratikte %20 ile %50 arasında dropout uygulandığı görülüyor.

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) #Softmax katmanı ile çıktı değerlerimizin hepsini bir olasılık dağılımı olarak hesaplanmasını sağlıyoruz.

model.summary()

# Ağ derlenirken kayıp fonksiyonunuzu ve optimizatörünüzü belirtmemiz isteniyor.
# Optimizasyon tiplerinden "RMSprop" ve yitim (loss) fonksiyonu olarak "categorical_crossentropy" kullanıyoruz.
# Son parametre olarak da hali hazırda tanımlanmış tek metrik fonksiyonu olan "accuracy" yi kullanıyoruz. Accuracy bize 0-1 arasında bir doğruluk değeri verecektir.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Burası en eğlenceli kısım, ağımızı eğitme kısmı :)
# Daha önce yüklenmiş eğitim verileri ile sınıflandırmayı öğrenmeye çalışır.
# Fit fonksiyonu eğitim sırasındaki yitim(loss)/doğrulama başarımı (accuracy) değerleri ile bir çok ayrıntı döndürür.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Son olarak eğitilmiş ağımızın test setimiz üzerindeki performans değerlerimizi hesaplayarak ekrana yazdıralım.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


