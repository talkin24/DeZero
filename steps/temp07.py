# -*- coding: utf-8 -*-
from sys import path

from numpy.core.defchararray import array


if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인
    import os
    import sys
    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


x = np.array([1, 2, 3])
np.save('test.npy', x)

x = np.load('test.npy')
print(x)


x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
data = {'x1': x1, 'x2': x2}

np.savez('test.npz', **data)

arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)
print(x2)
