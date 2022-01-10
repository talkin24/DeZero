# -*- coding: utf-8 -*-

import numpy as np
from dezero.core import Parameter
import dezero.functions as F
import weakref


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        # 출력이 하나 뿐이라면 튜플이 아니라 그 출력을 직접 반환
        self.outputs = [weakref.ref(y) for y in inputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):  # 자식 클래스에서 구현
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]  # layer 인스턴스에 담긴 parameter 인스턴스 꺼내기

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()  # 모든 매개변수에 대해 cleargrad 수행


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:  # in_size가 정해져 있지 않다면 나중으로 연기
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
