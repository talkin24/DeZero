# -*- coding: utf-8 -*-
import numpy as np


class Variable:
    def __init__(self, data):
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
        output = Variable(y)
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


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
