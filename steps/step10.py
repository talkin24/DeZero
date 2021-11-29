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
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)  # backward 메서드를 호출한다.

            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # 출력 변수에 창조자를 설정
        self.input = input  # 입력변수를 기억
        self.output = output  # 출력도 저장
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
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


# x = Variable(np.array(1.0))  # OK
# x = Variable(None)  # OK
# x = Variable(1.0)  # 오류 발생!


x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)
print(type(y))
