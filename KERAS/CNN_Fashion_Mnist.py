#Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.
#Amaç: Kıyafetlerin tanımlanması
#Veriseti: Fashion-Mnist
#Algoritma : Evrişimli Sinir Ağları (Convolutional Neural Networks)
#Hazırlayan: Can UMAY

#Gerekli kütüphanelerimizi içeri aktarıyoruz.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical

#Eğitim ve test verilerimizi aktarıyoruz.
(inp_train, out_train),(inp_test, out_test)=fashion_mnist.load_data()
#Inp_ olarak tanımlanan değişkenlerimizin boyutlarını düzenliyoruz.
inp_train=inp_train.reshape(-1,28,28,1)
inp_test=inp_test.reshape(-1,28,28,1)
#Daha sonra ondalık hale çeviriyoruz.
inp_train=inp_train.astype('float32')
inp_test=inp_test.astype('float32')
#Modelimizin daha optimize çalışması için değerlerimizi 0 ile 1 arasına indirgiyoruz.
inp_train=inp_train/255.0
inp_test=inp_test/255.0
#Out_ olarak tanımlanan değişkenleri ise one-hot-encoding haline getiriyoruz.
out_train=to_categorical(out_train)
out_test=to_categorical(out_test)

#Modelimizi oluşturmaya başlıyoruz.
model=Sequential()
# 3x3'lük 32 filtreli ve relu aktivasyon fonksiyonlu ilk Conv2D katmanımızı oluşturuyoruz.
model.add(Conv2D(32,(3,3),input_shape=(28,28,1), activation='relu'))
# 3x3'lük 32 filtreli ve relu aktivasyon fonksiyonlu ikinci Conv2D katmanımızı oluşturuyoruz.
model.add(Conv2D(32, (3,3), activation='relu'))
# 2x2 pool size'ı bulunan MaxPooling2D işlemi gerçekleştiriyoruz.
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten ile modelimizin Fully Connected kısmına bağlıyoruz.
model.add(Flatten())
# Fully Connected bölümünün ilk katmanını oluşturuyoruz.
model.add(Dense(64, activation='relu'))
# Son katmanımızı 10 nöron ile oluşturuyoruz çünkü 10 sınıfımız var.
model.add(Dense(10, activation='softmax'))

#Modeli compile edelim.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Artık modeli eğitebiliriz.
model.fit(inp_train,
          out_train,
          verbose=1,
          epochs=15,
          validation_split=0.2)

#Correction olarak oluşturduğumuz değişken ile modelimizin doğruluk oranını ölçelim.
correction=model.evaluate(inp_test.reshape(-1,28,28,1),out_test, verbose=1)
print('Yitim değeri (loss): {}'.format(correction[0]))
print('Test başarısı (accuracy): {}'.format(correction[1]))