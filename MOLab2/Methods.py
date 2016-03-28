import Representation
import sympy
from scipy import integrate
from sympy import Matrix
import numpy as np
import pylab

#c1 = 0, c2 = 0, c3 = 0, because we use problem for method of Ritz

a = 1; b = 2; b1 = 2; b2 = 1; b3 = 1; c1 = 0; c2 = 0; c3 = 0;
d1 = 1; d2 = 3; d3 = 1; k1 = 0; k2 = 2; p1 = 0; p2 = 2; q1 = 1; q2 = 0;
a1 = 1; a2 = 2; a3 = 4; a4 = 5; n1 = 5; n2 = 6; n3 = 1;

class BasicFunction:
    def __init__(self, k, a, b, N):
        self.k = k; self.a = a; self.b = b;
        self.N = N
    def __call__(self, x):
        h = (self.b - self.a) / self.N
        if self.k == 0:
            x1 = self.a + h
            if x >= self.a and x <= x1:
                return (x1 - x) / h
            else:
                return 0
        elif self.k == self.N:
            xx = self.b - h
            if x >= xx and x <= self.b:
                return (x - xx) / h
            else:
                return 0
        else:
            x_prev = self.a + (self.k - 1) * h
            x_cur = x_prev + h
            x_next = x_cur + h
            if x >= x_prev and x <= x_cur:
                return (x - x_prev) / h
            elif x > x_cur and x <= x_next:
                return (x_next - x) / h
            else:
                return 0

def DerBasicFunc(function, x):#derivative of the basic function
    h = (function.b - function.a) / function.N
    if function.k == 0:
        x1 = function.a + h
        if x >= function.a and x <= x1:
            return -1 / h
        else:
            return 0
    elif function.k == function.N:
        xx = function.b - h
        if x >= xx and x <= function.b:
            return 1 / h
        else:
            return 0
    else:
        x_prev = function.a + (function.k - 1) * h
        x_cur = x_prev + h
        x_next = x_cur + h
        if x >= x_prev and x <= x_cur:
            return 1 / h
        elif x > x_cur and x <= x_next:
            return -1 / h
        else:
            return 0

class Solution:
    def __init__(self, a, b, *coef):
        self.c = []
        self.f = []
        self.n = len(coef)
        for i in range(len(coef)):
            self.c.append(coef[i])
            self.f.append(BasicFunction(i, a, b, self.n - 1))
        #print([self.f[i].k for i in range(self.n)])
        #print(self.f[self.n - 1](b))
        #print("LAST COEF: " + str(self.c[self.n - 1]))
        #print("LENGTH OF BASE_F: " + str(len(self.f)))
    def __call__(self, x):
        s = sum([self.c[i] * self.f[i](x) for i in range(self.n)])
        return s

def FiniteElements(Problem, n):
    A = Matrix.zeros(n + 1, n + 1)
    b = Matrix.zeros(n + 1, 1)
    base_f = [BasicFunction(i, Problem.a, Problem.b, n) for i in range(n + 1)]
    alpha1 = Problem.Operator.k(Problem.a) * Problem.beta / Problem.alpha
    alpha2 = Problem.Operator.k(Problem.b) * Problem.delta / Problem.gamma
    h = (Problem.b - Problem.a) / n
    for i in range(0, n + 1):
        x_prev = Problem.a + (i - 1) * h
        x_cur = x_prev + h
        x_next = x_cur + h
        #A[i,i-1]
        if i != 0:
            I1 = lambda x: Problem.Operator.k(x)
            I2 = lambda x: Problem.Operator.q(x) * (x - x_prev) * (x_cur - x)
            res1 = integrate.quad(I1, x_prev, x_cur, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_prev, x_cur, epsabs = 1e-8)
            A[i,i - 1] = -1/h**2 * res1[0] + 1/h**2 * res2[0]
        #A[i,i+1]
        if i != n:
            I1 = lambda x: Problem.Operator.k(x)
            I2 = lambda x: Problem.Operator.q(x) * (x_next - x) * (x - x_cur)
            res1 = integrate.quad(I1, x_cur, x_next, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_cur, x_next, epsabs = 1e-8)
            A[i,i + 1] = -1/h**2 * res1[0] + 1/h**2 * res2[0]
        #A[i,i]
        if i == 0:
            I1 = lambda x: Problem.Operator.k(x)
            I2 = lambda x: Problem.Operator.q(x) * (x_next - x)**2
            res1 = integrate.quad(I1, x_cur, x_next, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_cur, x_next, epsabs = 1e-8)
            A[i, i] = 1/h**2 * res1[0] + 1/h**2 * res2[0] + alpha1 * (x_next - x_cur)**2 / h**2
        elif i == n:
            I1 = lambda x: Problem.Operator.k(x)
            I2 = lambda x: Problem.Operator.q(x) * (x - x_prev)**2
            res1 = integrate.quad(I1, x_prev, x_cur, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_prev, x_cur, epsabs = 1e-8)
            A[i, i] = 1/h**2 * res1[0] + 1/h**2 * res2[0] + alpha2 * (x_cur - x_prev)**2 / h**2
        else:
            I1 = lambda x: Problem.Operator.k(x)
            I2 = lambda x: Problem.Operator.q(x) * (x - x_prev)**2
            I3 = lambda x: Problem.Operator.q(x) * (x_next - x)**2
            res1 = integrate.quad(I1, x_prev, x_next, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_prev, x_cur, epsabs = 1e-8)
            res3 = integrate.quad(I3, x_cur, x_next, epsabs = 1e-8)
            A[i, i] = 1/h**2 * res1[0] + 1/h**2 * res2[0] + 1/h**2 * res3[0]
    for i in range(0, n + 1):
        x_prev = Problem.a + (i - 1) * h
        x_cur = x_prev + h
        x_next = x_cur + h
        if i == 0:
            I1 = lambda x: Problem.RightPart(x) * (x_next - x)
            res = integrate.quad(I1, x_cur, x_next, epsabs = 1e-8)
            b[i, 0] = 1/h * res[0]
        elif i == n:
            I1 = lambda x: Problem.RightPart(x) * (x - x_prev)
            res = integrate.quad(I1, x_prev, x_cur, epsabs = 1e-8)
            b[i, 0] = 1/h * res[0]
        else:
            I1 = lambda x: Problem.RightPart(x) * (x - x_prev)
            I2 = lambda x: Problem.RightPart(x) * (x_next - x)
            res1 = integrate.quad(I1, x_prev, x_cur, epsabs = 1e-8)
            res2 = integrate.quad(I2, x_cur, x_next, epsabs = 1e-8)
            b[i, 0] = 1/h * res1[0] + 1/h * res2[0]
        
    result = TDMAsolver(A, b)
    sol = Solution(Problem.a, Problem.b, *result)
    return sol

def TDMAsolver(A, b):#Tridiagonal Matrix Algorithm solver
    n = A.rows
    alpha = [A[0,1] / A[0,0]]
    beta = [b[0,0] / A[0,0]]
    x = [0] * n

    for i in range(1,n):
        if i != n - 1: alpha.append(A[i, i + 1] / (A[i,i] - alpha[i - 1] * A[i, i - 1]))
        beta.append((b[i] - beta[i - 1] * A[i, i - 1]) / (A[i, i] - alpha[i - 1] * A[i, i - 1]))

    x[n - 1] = beta[n - 1]
    for i in reversed(range(n - 1)):
        x[i] = beta[i] - alpha[i] * x[i + 1]

    return x

def MakeGraphic(function, left, right, num, xlabel, ylabel, title, Label, color):
    x = np.linspace(left, right, num)
    y = np.zeros(x.size)
    for i in range(x.size):
        y[i] = function(x[i])
    pylab.xlim(left, right)
    pylab.ylim(-200, 200)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.plot(x, y,'-' + color, label = Label)
    pylab.legend(loc='upper left')

Problem = Representation.Problem(a,b,b1,b2,b3,c1,c2,c3,d1,d2,d3,k1,k2,p1,p2,q1,q2,a1,a2,a3,a4,n1,n2,n3)
num = 100
sol = FiniteElements(Problem, num)

file = open("Output.txt", 'w')
h = (Problem.b - Problem.a) / num
file.write("xi".ljust(20) + "|" + "yi".ljust(20) + "|" + "U(xi)".ljust(20) + "|" + "yi - U(xi)".ljust(20) + '\n')
file.flush()
for i in range(num + 1):
    xi = Problem.a + i * h
    file.write(str(xi).ljust(20) + "|" + str(sol(xi)).ljust(20) + "|" + str(Problem.ExactSolution(xi)).ljust(20) + "|" + str(sol(xi) - Problem.ExactSolution(xi)).ljust(20) + '\n')
    file.flush()

left = a
right = b
MakeGraphic(sol, left, right, num, 'x','y', 'Laboratorna 1', 'Approximate solution (finite elements)', 'r')
left = a - 5
right = b + 5
MakeGraphic(Problem.ExactSolution, left , right, 5000, 'x','y', 'Laboratorna 1', 'Exact solution', 'b')
pylab.show()
