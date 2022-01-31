# -*- coding: utf-8 -*-
from sys import path
import os
import sys


if '__file__' in globals():  # __file__ 이라는 전역변수가 정의되어 있는지 확인

    # 현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero.functions as F
from dezero.models import VGG16
from PIL import Image
import dezero

url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])