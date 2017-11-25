'''
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.

Amaç: El yazısı rakamların tanınması.
Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)

10 epoch sonunda testde 98% doğruluk oranı elde edilmiştir.

Nasıl çalıştırılır ?
python main.py
CUDA_VISIBLE_DEVICES=2 python main.py  # İstenilen GPU'da çalıştırmak için.
'''

# Gerekli kütüphanelerin dahil edilmesi:
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Terminal komutundan alınan bilginin işlenmesi:
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() # Cuda var mı diye kontrol edilir.

# Rastgele sayı üretmek için:
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# MNIST verisetini içe aktarılması:
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Evrişimli Sinir Ağları modelinin oluşturulması:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Giriş kanalı: 1, Çıkış kanalı: 10, Filtre boyutu: 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Giriş kanalı: 10, Çıkış kanalı: 20, Filtre boyutu: 5x5

        # Rastgele olacak şekilde nöronların %50'sini kapatıyoruz (modelin ezberlemesini engeller):
        self.conv2_drop = nn.Dropout2d() # Fonksiyonun varsayılan kapama oranı %50

        self.fc1 = nn.Linear(320, 50) # Giriş nöron sayısı: 320, Çıkış nöron sayısı: 50
        # Modele yeni bir katmanda 50 nöron eklemiş olduk.

        self.fc2 = nn.Linear(50, 10) # Giriş nöron sayısı: 50, Çıkış nöron sayısı: 10
        # 10 sınıfımızı temsil edecek 10 nöron.

    # Modelin akış şemasını oluşturalım:
    def forward(self, x):
        # Giriş(x) boyutu: [1, 28, 28] x 64(batch_size) Kanal syaısı: 1, Görselin boyutu: 28x28

        # Girişi, yukarıda tanımladığımız "conv1" katmanından geçiriyoruz,
        # sonra 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı ekliyoruz,
        # daha sonra ReLu aktivasyon fonksiyonumuzdan geçiriyoruz:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Çıkış boyutu: [10, 12, 12]

        # Yukarıda aldığımız çıktıyı "conv2" katmanından geçiriyoruz,
        # sonra yukarıda tanımladığımız Dropout katmanımızı ekliyoruz,
        # daha sonra 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı uyguluyoruz,
        # en sonda ReLu aktivasyon fonksiyonumuzdan geçiriyoruz:
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Çıkış boyutu: [20, 4, 4]

        x = x.view(-1, 320) # Yeniden boyutlandırma yapıyoruz.
        # 4x4 boyutlu 20 kanallı fotoğrafı 1 boyutlu vectöre çeviriyoruz.
        # -1 boyutu giriş boyutuna ve belirlenen diğer boyutlara bakılarak bulunur.
        # 20x4x4 = 320:
        # Çıkış boyutu: [320]

        # Yukarıda tanımladığımız "fc1" katmanımızdaki 50 nöronu modelimize ekliyoruz,
        # daha sonra çıktımızı ReLu aktivasyon fonksiyonumuzdan geçiriyoruz:
        x = F.relu(self.fc1(x))
        # Çıkış boyutu: [50]

        # Modelin ezberlemesini önlemek için Dropout katmanımızı ekliyoruz:
        x = F.dropout(x, training=self.training)

        # Yukarıda tanımladığımız "fc2" katmanımızdaki 10 nöronu modelimize ekliyoruz,
        # daha sonra çıktımızı ReLu aktivasyon fonksiyonumuzdan geçiriyoruz:
        x = self.fc2(x)
        # Çıkış boyutu: [10]
        # Verisetimizdeki 10 sınıfı temsil edecek 10 çıktıyı elde ettik.

        # Son olarak sınıflandırma yapmak için Softmax fonksiyoumuzu kullanıyoruz:
        return F.log_softmax(x)

model = Net() # Modelimizi tanımlıyoruz.
if args.cuda:
    model.cuda() # Verileri GPU'ya taşır.

# "SGD" optimizasyon fonksiyonumuzu oluşturuyoruz:
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Modeli eğitecek fonksiyonumuzu oluşturuyoruz:
def train(epoch):
    model.train() # Modelimizi eğitim moduna alıyoruz.
    for batch_idx, (data, target) in enumerate(train_loader): # Verisetini batch'lere bölüyoruz.
        if args.cuda:
            data, target = data.cuda(), target.cuda() # Verileri GPU'ya taşır.
        data, target = Variable(data), Variable(target) # Verilerimizi PyTorch değişkenlerine(Tensor) çeviriyoruz.
        optimizer.zero_grad() # Tüm optimize edilmiş değişkenlerin verilerini temizler.
        output = model(data) # Girdi verisini modelimizde işliyoruz ve çıktımızı alıyoruz.
        # Çıkması gereken sonuç ile modelimizin ürettiği çıktıyı karşılaştırarak hata hesaplamamızı yapıyoruz:
        loss = F.nll_loss(output, target) # Hata fonksiyonumuz: The negative log likelihood loss(NLLLoss)
        loss.backward() # Bulduğumuz hata oranıyla geri-yayılım uyguluyoruz.
        optimizer.step() # Modelimizi(ağırlıkları) daha optimize sonuç için güncelliyoruz.

        # Belli aralıklarla(log_interval, varsayılan değer: 10) modelin başarını ekrana yazdırıyoruz:
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

# Modeli test edecek fonksiyonumuzu oluşturuyoruz:
def test():
    model.eval() # Modeli test moduna alıyoruz.
    test_loss = 0
    correct = 0
    for data, target in test_loader: # Test verimizi alıyoruz.
        if args.cuda:
            data, target = data.cuda(), target.cuda() # Verileri GPU'ya taşır.
        data, target = Variable(data, volatile=True), Variable(target) # Verilerimizi PyTorch değişkenlerine(Tensor) çeviriyoruz.
        output = model(data) # Girdi verisini modelimizde işliyoruz ve çıktımızı alıyoruz.
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # Batch hata oranının hesaplanması ve toplam hata oranına eklenmesi.
        # Çıkması gereken sonuç ile modelimizin ürettiği çıktıyı karşılaştırarak hata hesaplamamızı yapıyoruz:
        pred = output.data.max(1)[1] # Maksimim olasılık indeksi alınarak sonuç elde edilir.
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # Modelin başarısı hesaplanır.

    # Modelin başarısını ekrana yazdırıyoruz:
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Eğitimi başlatıyoruz:
for epoch in range(1, args.epochs + 1):
    train(epoch) # Model eğitilir.
    test() # Model test edilir.
