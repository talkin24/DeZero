# -*- coding: utf-8 -*-
from sys import path


if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os
    import sys
    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 1. setting Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 2. read data / creat model, optimizer
train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 3. shuffle idx in dataset
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 4. create minibatch
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        # 5. calculate gradient / update parameters
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # 6. print training process each epoch
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
