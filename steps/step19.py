# -*- coding: utf-8 -*-
import numpy as np
import unittest
import weakref
import contextlib


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data  # 통상값
        self.name = name
        self.grad = None  # 미분값
        self.creator = None
        self.generation = 0  # 세대를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대를 기록(부모세대 + 1)

    def backward(self, retain_grad=False):
        if self.grad is None:
            # self.data와 형상과 데이터 타입이 같은 ndarray 인스턴스 생성
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 가져온다
            # 변경 전: gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조(weakref)

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        # 줄바꿈아 있으면 줄바꿈 뒤에 간 공백을 넣어 숫자의 시작 위치가 가지런 하게 표시되게 함
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 리스트 언팩
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 튜플로 반환
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 세대 정보는 역전파 시에만 사용. creator 설정 역시 역전파 시 필요 없음
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # 입력변수를 기억
            self.outputs = [weakref.ref(output)
                            for output in outputs]  # 출력변수를 약하게 참조

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y  # 튜플로 반환

    def backward(self, gy):  # 입력이 1개, 출력이 2개
        return gy, gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy  # gy는 출력쪽에서 전해지는 미분값
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class SquareTest(unittest.TestCase):  # unittest.TestCase를 상속
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():  # 순전파만 사용하고 싶을때
    return using_config('enable_backprop', False)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.shape)
print(len(x))
print(x)
