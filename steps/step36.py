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


def f(x):
    t = x ** 2
    y = F.sum(t)
    return y


x = Variable(np.array([1.0, 2.0]))
v = Variable(np.array([4.0, 5.0]))

y = f(x)
y.backward(creat_graph=True)

gx = x.grad
x.cleargrad()

z = F.matmul(v, gx)
z.backward()
print(x.grad)
