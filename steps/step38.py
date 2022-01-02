# -*- coding: utf-8 -*-
from sys import path


if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os
    import sys
    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6, ))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
print(x)
y = x.reshape((2, 3))
print(y)
y = x.reshape(2, 3)
print(y)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)


x = Variable(np.random.randn(2, 3))
y1 = x.transpose()
print(y1)
print(type(y1))
y2 = x.T
print(y2)
print(type(y2))
print(y1 != y2)

A, B, C, D = 1, 2, 3, 4
x = np.random.randn(A, B, C, D)
print(x)
print(x.shape)
y = x.transpose(1, 0, 3, 2)
print(y)
print(y.shape)

print('*'*100)
x = Variable(np.random.randn(2, 3, 4))
y0 = F.transpose(x)
print(y0.shape)
y1 = F.transpose(x, axes=(1, 0, 2))
print(y1.shape)
