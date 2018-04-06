
###IMDB duygusallıklarının sınıflandırılması için eğitimi tekrarlayan sinir ağı modellemesi###
###Teorik olarak elde ettiğimiz verileri ilk olarak kelime değerlerini tek boyutlu vektöre çevireceği###
###Vektörler teorik olarak 0 ve 1 ile oluşacak ve bol miktarda 0 oluşmasına karşı embedding işlemi uygulanacaktır##
###İşlem sonrası elde edilen oldukça uzun vektörü ise 4 birimlik vektör haline dönüştüreceğiz##


from __future__ import print_function

from keras.preprocessing import sequence ##import edilen verilerin sıralı bir şekilde ön işleme tutulmasını sağlar ##
from keras.models import Sequential ## kerasın ardışık modellemesini import eder##
from keras.layers import Dense, Dropout, Activation 
from keras.layers import Embedding##Embedding işlemi için embedding katmanınını import ettik##
from keras.layers import LSTM ## lstm katmanını import ettik###
from keras.layers import Conv1D, MaxPooling1D###Verilerimiz tek boyutlu vektör olarak işleme alacağımız için pooling ve CNN işlemlerinin tek boyutlu katmanlarını import ettik### 
from keras.datasets import imdb ## keras datasetinden imdb verileri import edilir##

# Embedding
max_features = 20000##Maximum kelime hafızası sayısı##
maxlen = 100##Kelimelerin maximum uzunlukları
embedding_size = 128##Kelime vektörlerine uygulanacak embedding boyutu###

# Convolution
kernel_size = 5 ##Sinir ağımızın çekirdek sayısını belirledik###
filters = 64###Sinir ağımızda vektöre ne kadarlık filtre uygulanacağını belirledik###
pool_size = 4##verilerden elde ettiğimiz matrisin boyutlandırma miktarı##

# LSTM
lstm_output_size = 70##lstm sonucundaki çıktıların boyutlarını belirler##

# Training
batch_size = 30##her döngüde kaç veri işleneceğini belirtir##
epochs = 2##eğitimin kaç adımda gerçekleşeceğini belirtir##

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')##İmport edilen veri setini yazdırır##
##import edilen veri setinden eğitim ve test verilerinin belirlenmesi##  
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train),)##eğitim dizlerinin vektörel uzunluğunu yazdırır##
print(len(x_test), )## test dizilerinin vektörel uzunluğunu yazdırır##

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')

model = Sequential()###model olarak sıralı modelimizi seçtik###
model.add(Embedding(max_features, embedding_size, input_length=maxlen))###Modelimize embedding işlemini ekledik###
model.add(Dropout(0.25))## ağ modellemesindeki nöronların % kaç kadar kapatılacağını gösterir## 
model.add(Conv1D(filters,###verileri tek boyutlu bir şekilde işler###
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))##verilerden elde edilen matrisi boyutlandırır##
model.add(LSTM(lstm_output_size))##Modelimize LSTM'i ekledik##
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')##modelin eğitime başladığını gösterir##
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)##test sonucunu gösterir##
print('Test accuracy:', acc)##test sonucundaki doğruluk sonucunu gösterir##
