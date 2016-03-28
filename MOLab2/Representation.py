from sympy import *

class RightPart:
    def __init__(self, a = 1, b = 4, b1 = 2, b2 = 1, b3 = 1, c1 = 2, c2 = 1, c3 = 1, d1 = 1, d2 = 3, d3 = 1, k1 = 0, k2 = 2, p1 = 0, p2 = 2, q1 = 1, q2 = 0,
                 a1 = 1, a2 = 2, a3 = 4, a4 = 5, n1 = 5, n2 = 3, n3 = 1):
        self.a = a; self.b = b; self.b1 = b1; self.b2 = b2; self.b3 = b3; self.c1 = c1; self.c2 = c2; self.c3 = c3;
        self.d1 = d1; self.d2 = d2; self.d3 = d3; self.k1 = k1; self.k2 = k2; self.p1 = p1; self.p2 = p2; self.q1 = q1; self.q2 = q2;
        self.a1 = a1; self.a2 = a2; self.a3 = a3; self.a4 = a4; self.n1 = n1; self.n2 = n2; self.n3 = n3;
    def __call__(self, x):
        if x == 0.0:
            return 32.0
        result = (-(self.b1 * x**self.k1 + self.b2 * x**self.k2 + self.b3) * (self.a1 * self.n1 * (self.n1 - 1) * x**(self.n1 - 2) +
                                                                             self.a2 * self.n2 * (self.n2 - 1) * x**(self.n2 - 2) +
                                                                             self.a3 * self.n3 * (self.n3 - 1) * x**(self.n3 - 2))
                 -(self.b1 * self.k1 * x**(self.k1 - 1) + self.b2 * self.k2 * x**(self.k2 - 1)) * (self.a1 * self.n1 * x**(self.n1 - 1) +
                                                                                                   self.a2 * self.n2 * x**(self.n2 - 1) +
                                                                                                   self.a3 * self.n3 * x**(self.n3 - 1))
                 +(self.c1 * x**self.p1 + self.c2 * x**self.p2 + self.c3) * (self.a1 * self.n1 * x**(self.n1 - 1) +
                                                                             self.a2 * self.n2 * x**(self.n2 - 1) +
                                                                             self.a3 * self.n3 * x**(self.n3 - 1))
                 +(self.d1 * x**self.q1 + self.d2 * x**self.q2 + self.d3) * (self.a1 * x**self.n1 +
                                                                             self.a2 * x**self.n2 +
                                                                             self.a3 * x**self.n3 + self.a4))
        return result

class Operator:
    def __init__(self, a = 1, b = 4, b1 = 2, b2 = 1, b3 = 1, c1 = 2, c2 = 1, c3 = 1, d1 = 1, d2 = 3, d3 = 1, k1 = 0, k2 = 2, p1 = 0, p2 = 2, q1 = 1, q2 = 0,
                 a1 = 1, a2 = 2, a3 = 4, a4 = 5, n1 = 5, n2 = 3, n3 = 1):
        self.a = a; self.b = b; self.b1 = b1; self.b2 = b2; self.b3 = b3; self.c1 = c1; self.c2 = c2; self.c3 = c3;
        self.d1 = d1; self.d2 = d2; self.d3 = d3; self.k1 = k1; self.k2 = k2; self.p1 = p1; self.p2 = p2; self.q1 = q1; self.q2 = q2;
        self.a1 = a1; self.a2 = a2; self.a3 = a3; self.a4 = a4; self.n1 = n1; self.n2 = n2; self.n3 = n3;
    def k(self, x):
        return self.b1 * x**self.k1 + self.b2 * x**self.k2 + self.b3
    def p(self, x):
        return self.c1 * x**self.p1 + self.c2 * x**self.p2 + self.c3
    def q(self, x):
        return self.d1 * x**self.q1 + self.d2 * x**self.q2 + self.d3
    def __call__(self, function, x):
        arg = Symbol('x')
        print(x)
        print(arg)
        k1 = diff(self.k(arg), arg).subs(arg, x)
        print("K1: " + str(k1))
        print(x)
        print(arg)
        u1 = diff(function(arg), arg).subs(arg, x)
        print("U1: " + str(u1))
        u2 = diff(function(arg), arg, 2).subs(arg, x)
        print("U2: " + str(u2))
        return -self.k(x) * u2 + (self.p(x) - k1) * u1 + self.q(x) * function(x)

class ExactSolution:
    def __init__(self, a1 = 1, a2 = 2, a3 = 4, a4 = 5, n1 = 5, n2 = 3, n3 = 1):
                 self.a1 = a1; self.a2 = a2; self.a3 = a3; self.a4 = a4; self.n1 = n1; self.n2 = n2; self.n3 = n3;
    def __call__(self, x):
        return self.a1 * x**self.n1 + self.a2 * x**self.n2 + self.a3 * x**self.n3 + self.a4

class Problem:
    def __init__(self, a = 1, b = 4, b1 = 2, b2 = 1, b3 = 1, c1 = 2, c2 = 1, c3 = 1, d1 = 1, d2 = 3, d3 = 1, k1 = 0, k2 = 2, p1 = 0, p2 = 2, q1 = 1, q2 = 0,
                 a1 = 1, a2 = 2, a3 = 4, a4 = 5, n1 = 5, n2 = 3, n3 = 1):
        self.Operator = Operator(a,b,b1,b2,b3,c1,c2,c3,d1,d2,d3,k1,k2,p1,p2,q1,q2,a1,a2,a3,a4,n1,n2,n3)
        self.RightPart = RightPart(a,b,b1,b2,b3,c1,c2,c3,d1,d2,d3,k1,k2,p1,p2,q1,q2,a1,a2,a3,a4,n1,n2,n3)
        self.ExactSolution = ExactSolution(a1,a2,a3,a4,n1,n2,n3)
        self.a = a; self.b = b;
        self.alpha = a1 * a**n1 + a2 * a**n2 + a3 * a**n3 + a4
        self.beta = a1 * n1 * a**(n1 - 1) + a2 * n2 * a**(n2 - 1) + a3 * n3 * a**(n3 - 1)
        self.gamma = a1 * b**n1 + a2 * b**n2 + a3 * b**n3 + a4
        self.delta = -(a1 * n1 * b**(n1 - 1) + a2 * n2 * b**(n2 - 1) + a3 * n3 * b**(n3 - 1))
