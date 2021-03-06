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


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
print('ADD', '*'*100)
y = x0 + x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

print('MUL', '*'*100)
y = x0 * x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

print('SUB', '*'*100)
y = x0 - x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

print('DIV', '*'*100)
y = x0 / x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)
