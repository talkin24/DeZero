# -*- coding: utf-8 -*-
from sys import path


if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os
    import sys
    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


# 하이퍼파라미터 설정
lr = 0.2
max_iters = 10000
hidden_size = 10

# 모델 정의
model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)
# 또는 optimizer.SGD(lr).setup(model)

# 학습 시작
for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)
