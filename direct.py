import numpy as np
import sys
import math

args = sys.argv
class Rectangle:
    def __init__(self, fun, c, e):
        self.emin = min(e)  # Smallest exponent/longest side length/index of rectangle set
        self.dmax = (1/3)**self.emin
        self.index = 0  # Index of this rectangle in the array
        self.c = c  # Keep a copy of the center of the rectangle
        self.e = e  # Keep a copy of the edge lengths of the rectangle, l=(1/3)^e
        if isinstance(fun, float):
            self.fc = fun
        else:
            self.fc = fun(c)  # Evaluate function at the center of the rectangle

# Example usage
def opt_fun(x):
    return x**2

r = Rectangle(opt_fun, 0, [1, 2, 3])

class new_rectangle:
    def __init__(self, fun, c, e):
        self.emin = min(e)  # 最小指数/最長辺の長さ/長方形集合のインデックス
        self.dmax = (1/3) ** self.emin  # 最大辺長
        self.index = 0  # 長方形配列内のこの長方形のインデックス
        self.c = c  # 長方形の中心をコピーする
        self.e = e  # 長方形の辺の長さを保持する、l=(1/3)^e
        if isinstance(fun, float):
            self.fc = fun
        else:
            self.fc = fun(c)  # 長方形の中心で関数を評価する

def object_function():
    pass#Sphere function

# dimention
D = 4

opt_fun = lambda x: object_function(x)
#Create initial rectangle
c = [0.5 for i in range(D)]
e = [0 for i in range(D)]
r = new_rectange(opt_fun, c, e)

max_num_rectangles = 1000
epsilon = 1e-10
emax = int(math.ceil(math.log(epsilon) / math.log(1 / 3)))
enum = emax + 1
elist = [i for i in range(enum)]
dlist = [(1 / 3)**i for i in elist]
print('epsilon = {}, emax = {}, (1/3)^emax = {}'.format(epsilon, emax, (1 / 3)**emax))
points = [None for i in range(enum)]
points[r.emin + 1] = [r]

# kotaisu
N = 2
iter = 200

if args[1] == 'sphere':

for i in range(1, iter):
    if len(points) > enum:
        break
    for j in range(2, len(points))


