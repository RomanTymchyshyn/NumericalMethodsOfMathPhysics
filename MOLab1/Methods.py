import Representation
import sympy
from scipy import integrate
from sympy import Matrix
import numpy as np
import pylab


a = 1; b = 2; b1 = 2; b2 = 1; b3 = 1; c1 = 2; c2 = 1; c3 = 1;
d1 = 1; d2 = 3; d3 = 1; k1 = 0; k2 = 2; p1 = 0; p2 = 2; q1 = 1; q2 = 0;
a1 = 1; a2 = 2; a3 = 4; a4 = 5; n1 = 5; n2 = 6; n3 = 1;

def ScalarMultiply(Op1, f1, Op2, f2, a, b):
    """ Scalar multiply.

        First argument - operator which will apply to first operand (if 0 then no operator to apply)
        Second argument - first function
        Third argument - operator which will apply to second operand (if 0 then no operator to apply)
        Fourth argument - second function
    """
    if Op1 == 0:
        g1 = lambda x: f1(x)
    else:
            g1 = lambda x: Op1(f1, x)
    if Op2 == 0:
        g2 = lambda x: f2(x)
    else:
        g2 = lambda x: Op2(f2, x)
    g = lambda x: g1(x) * g2(x)
    result = integrate.quad(g, a, b, epsabs = 1e-8)#result = (result of integration, error)
    return result[0]

def Discrepancy(Problem, sol):
    f = lambda x: Problem.Operator(sol, x) - Problem.RightPart(x)
    expr = lambda x: f(x) * f(x)
    result = integrate.quad(expr, Problem.a, Problem.b, epsabs = 1e-8)#result = (result of integration, error)
    return result[0]**(1/2)

class BasicFunction:
    def __init__(self, k, a, b, alpha, beta, gamma, delta):
        self.k = k; self.a = a; self.b = b;
        self.A = b + gamma * (b - a) / (2 * gamma + delta * (b - a))
        self.B = a + alpha * (a - b) / (2 * alpha - beta * (a - b))
    def __call__(self, x):
        if self.k > 2:
            return (x - self.a)**(self.k - 1) * (x - self.b)**2
        elif self.k == 2:
            return (x - self.b)**2 * (x - self.B)
        elif self.k == 1:
            return (x - self.a)**2 * (x - self.A)
        else:
            return 0

class Solution:
    def __init__(self, a, b, alpha, beta, gamma, delta, *coef):
        self.c = []
        self.f = []
        self.n = len(coef)
        for i in range(len(coef)):
            self.c.append(coef[i])
            self.f.append(BasicFunction(i + 1, a, b, alpha, beta, gamma, delta))
    def __call__(self, x):
        s = sum([self.c[i] * self.f[i](x) for i in range(self.n)])
        return s
    
def LeastSquares(Problem, n):
    A = Matrix.zeros(n)
    base_f = [BasicFunction(i, Problem.a, Problem.b, Problem.alpha, Problem.beta, Problem.gamma, Problem.delta) for i in range(1, n + 1)]
    for i in range(n):
        for j in range(n):
            A[i,j] = ScalarMultiply(Problem.Operator, base_f[i], Problem.Operator, base_f[j], Problem.a, Problem.b)
    b = Matrix.zeros(n, 1)
    for i in range(n):
        b[i,0] = ScalarMultiply(0, Problem.RightPart, Problem.Operator, base_f[i], Problem.a, Problem.b)
    c = A.solve_least_squares(b)
    sol = Solution(Problem.a, Problem.b, Problem.alpha, Problem.beta, Problem.gamma, Problem.delta, *c)
    return sol

def Collocation(Problem, n):
    A = Matrix.zeros(n)
    base_f = [BasicFunction(i, Problem.a, Problem.b, Problem.alpha, Problem.beta, Problem.gamma, Problem.delta) for i in range(1, n + 1)]
    x_arr = [Problem.a + i * (Problem.b - Problem.a) / (n + 1) for i in range(1, n + 1)]
    for i in range(n):
        for j in range(n):
            A[i,j] = Problem.Operator(base_f[j],x_arr[i])
    b = Matrix.zeros(n, 1)
    for i in range(n):
        b[i,0] = Problem.RightPart(x_arr[i])
    c = A.LUsolve(b)
    sol = Solution(Problem.a, Problem.b, Problem.alpha, Problem.beta, Problem.gamma, Problem.delta, *c)
    return sol

def MakeGraphic(function, left, right, num, xlabel, ylabel, title, Label, color):
    x = np.linspace(left, right, num)
    y = function(x)
    pylab.xlim(left, right)
    pylab.ylim(-100, 100)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.plot(x, y,'-' + color, label = Label)
    pylab.legend(loc='upper right')

Problem = Representation.Problem(a,b,b1,b2,b3,c1,c2,c3,d1,d2,d3,k1,k2,p1,p2,q1,q2,a1,a2,a3,a4,n1,n2,n3)
sol1 = LeastSquares(Problem, 5)
sol2 = Collocation(Problem, 5)
print("||LU - f|| = " + str(Discrepancy(Problem, sol1)) + " (least squares)")
print("||LU - f|| = " + str(Discrepancy(Problem, sol2)) + " (Collocations)")
input("Please, press any key to resume")

MakeGraphic(Problem.ExactSolution, -5,5,1000, 'x','y', 'Laboratorna 1', 'Exact solution', 'b')
MakeGraphic(sol1, -5,5,1000, 'x','y', 'Laboratorna 1', 'Approximate solution (least square)', 'r')
MakeGraphic(sol2, -5,5,1000, 'x','y', 'Laboratorna 1', 'Approximate solution (Collocation)', 'g')
pylab.show()
