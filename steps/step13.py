# -*- coding: utf-8 -*-
import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data  # 통상값
        self.grad = None  # 미분값
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # def backward(self):
    #     f = self.creator  # 1. 함수를 가져온다.
    #     if f is not None:
    #         x = f.input  # 2. 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다.
    #         x.backward()  # 하나 앞 변수의 backward 메서드를 호출한다.(재귀)

    def backward(self):
        if self.grad is None:
            # self.data와 형상과 데이터 타입이 같은 ndarray 인스턴스 생성
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 함수를 가져온다
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx  # 역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 리스트 언팩
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 튜플로 반환
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs  # 입력변수를 기억
        self.outputs = outputs  # 출력도 저장

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


x = Variable(np.array(2))
y = Variable(np.array(3))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)
