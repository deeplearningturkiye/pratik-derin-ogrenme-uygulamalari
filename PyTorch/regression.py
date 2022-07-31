#!/usr/bin/env python

"""
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.
"""

from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

# Polinom Özellik Dönüşümleri
POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Özellikler oluşturur. Yani [x, x^2, x^3, x^4] kolonları olan bir matris."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Yaklaşık Fonksiyon."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Polinomun string olarak açıklamasını oluşturur."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Batch oluşturur. Yani bir (x, f(x)) çifti."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# Modelin Tanımlanması.
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # Veriyi Al.
    batch_x, batch_y = get_batch()

    # Gradyanları Sıfırla.
    fc.zero_grad()

    # Doğrudan Geç.
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # Geriye Dön.
    output.backward()

    # Gradyanları Uygula.
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad)

    # Durma Kriterlerini Belirle.
    if loss < 1e-3:
        break

# Kayıp
print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))

# Öğrenilmiş Fonksiyon
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))

# Asıl Fonksiyon
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))