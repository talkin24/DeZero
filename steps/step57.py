# -*- coding: utf-8 -*-
from sys import path
import os
import sys

if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인

    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero.functions as F


x1 = np.random.rand(1, 3, 7, 7)  # batch size = 1
col1 = F.im2col
