import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap

import random
from scipy.stats import cauchy

args = sys.argv
"""
上から、
DIRECT法 ------> UMAP/t-SNE ------> SHADE
となっている。

    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(19))
    plt.title('t-SNE')
    plt.show()

    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(19))
    plt.title('UMAP')
    plt.show()

"""


def object_fun(x):
    """
    Declares the objective function.
    :param x:[x0,...,xn]
    :return:the value of f(x) (float type)
    """
    if(args[1] == "sphere"):
        y = 0.
        for i in x:
            y += pow(i, 2)
        return float(y)
    elif(args[1] == "rastrigin"):
        y = 0.
        for i in x:
            y += np.power(i, 2) - (10*np.cos(2*np.pi*i))
        y += 10*D
        return float(y)
    elif(args[1] == "rosenblock"):
        y=0.
        for i in range(len(x)-1):
            t1 = 100*(x[i+1] -x[i] ** 2) ** 2
            t2 = (x[i-1])** 2
            y += t1 + t2
        return float(y)
def get_center_point(bands):
    """
    center pointを返す関数
    :param bands:探索範囲
    :return:中心点
    """
    D = len(bands)
    center_point = np.empty((D, 1))
    for i in range(D):
        center_point[i] = (bands[i][0] + bands[i][1]) / 2
    return center_point


def get_len_of_interval(interval):
    return interval[1] - interval[0]


def divide_one_to_tree_block(bands, divided_D):
    delta = get_len_of_interval(bands[divided_D]) / 3
    divided_bands_list = []
    divided_bands = bands.copy()
    divided_bands[divided_D][1] -= delta * 2
    divided_bands_list.append(divided_bands)

    divided_bands = bands.copy()
    divided_bands[divided_D][0] += delta
    divided_bands[divided_D][1] -= delta
    divided_bands_list.append(divided_bands)

    divided_bands = bands.copy()
    divided_bands[divided_D][0] += delta * 2
    divided_bands_list.append(divided_bands)

    return divided_bands_list


def get_max_side_dimension_list(bands):
    D = len(bands)
    max_side_length = np.longdouble(0)
    # I
    max_side_dimension_list = []
    for dimension in range(D):
        length = get_len_of_interval(bands[dimension])
        max_side_dimension_list.append(dimension)
    return max_side_dimension_list


def remove_array(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def get_divided_bands_list(f, bands, divided_points):
    max_side_dimension_list = get_max_side_dimension_list(bands)
    delta = get_len_of_interval(bands[max_side_dimension_list[0]]) / 3
    center_point = get_center_point(bands)

    def get_wi(dimension):
        """
        :param dimension: ある次元D_i
        :return: wi
        """
        a_point = center_point.copy()
        a_point[dimension] -= delta
        b_point = center_point.copy()
        b_point[dimension] += delta
        #print('D = {}, a_point = {}, b_point= {}'.format(dimension, a_point, b_point))
        fa = f(a_point)
        fb = f(b_point)
        wi = min(fa, fb)
        if(fa > fb):
            divided_points.append(b_point[dimension][0])
        else:
            divided_points.append(a_point[dimension][0])
        return wi

    sorted_max_side_dimension_list = sorted(max_side_dimension_list, key=get_wi)
    # 3つの空間に分割する
    divided_bands_list = [bands]
    divided_bands = bands
    for dimension in sorted_max_side_dimension_list:
        remove_array(divided_bands_list, divided_bands)
        divided_bands_list += divide_one_to_tree_block(divided_bands, dimension)
        divided_bands = divided_bands_list[-2]
    return divided_bands_list


def sort_points(point_array):
    def slope(y):
        x = point_array[0]
        if math.isclose(y[0], x[0]):
            return x[1] - y[1] / 0.00000000001

        return (x[1] - y[1]) / (x[0] - y[0])

    def k(point):
        return point[0]

    point_array.sort(key=k)  # put leftmost first
    point_array = point_array[:1] + sorted(point_array[1:], key=slope)
    return point_array


def graham_scan(point_array):
    """Takes an array of points to be scanned.
    Returns an array of points that make up the convex hull surrounding the points passed in in point_array.
    """
    def cross_product_orientation(a, b, c):
        """Returns the orientation of the set of points.
        >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
        """
        return (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])

    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    sorted_points = sort_points(point_array)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and cross_product_orientation(convex_hull[-2], convex_hull[-1], p) >= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return convex_hull


def select_bands(f, bands_list):
    points = []
    for bands in bands_list:
        center_point = get_center_point(bands)
        dis = 0.
        D = len(bands)
        for i in range(D):
            dis += math.pow(bands[i][0] - center_point[i], 2)
        dis = math.sqrt(dis)
        point = [dis, f(center_point), bands.copy()]
        points.append(point)
    hull = graham_scan(points)
    selected_bands = []
    for point in hull:
        selected_bands.append(point[2])
    return selected_bands

def DIRECT(f, bands, iter, num):
    """
    DIRECT implementation.
    :param f: the f(x) you need analysis. such as object_fun(x).
    :param bands:
    :param iter:
    :return: divided points
    """
    l = get_len_of_interval(bands[0]) / 2
    D = len(bands)

    center_point = get_center_point(bands)

    convert_bands = np.zeros((D, 2))
    for i in range(D):
        convert_bands[i] = [0, 1]

    def fun_convert(x):
        convert_x = []
        for i in range(D):
            convert_x.append(x[i] * 2 * l + center_point[i] - l)
        return f(convert_x)
    return half_DIRECT(fun_convert, convert_bands, iter, D, num)


def half_DIRECT(f, bands, iter, D, num):
    """
    DIRECT Algorithm [0,1]であることに注意
    Note that the range of bands is [0,1]

    :param f:fun_convert(x)
    :param bands:the search range for the unit hypercube
    :param iter:number iterations for the unit hypercube.
    :param D:dimenton
    :return min_f
    """
    # min_fの初期値
    center_point = get_center_point(bands)
    min_f = f(center_point)

    #初期状態では、bandsが唯一の立方体
    total_bands_list = [bands]
    pure_divided_bands_list = [bands]
    divide_points_list = []

    for i in range(iter):
        print('iter = {}, cnt = {}'.format(i, int(len(divide_points_list)/D)))
        if(num < int(len(divide_points_list)/D)):
            divide_points_list_np = np.array(divide_points_list)
            a = int(len(divide_points_list)/D)
            b = D
            convert_list = divide_points_list_np.reshape(a, b)
            return convert_list

        bands_list = select_bands(f, pure_divided_bands_list)
        pure_divided_bands_list = []
        for bands in bands_list:
            remove_array(total_bands_list, bands)
            pure_divided_bands_list += get_divided_bands_list(object_fun, bands, divide_points_list)
            for divided_bands in pure_divided_bands_list:
                center_point = get_center_point(divided_bands)
                min_f = min(f(center_point), min_f)
            total_bands_list += pure_divided_bands_list
    print(len(divide_points_list),num)
    divide_points_list_np = np.array(divide_points_list)
    a = int(len(divide_points_list)/D)
    b = D
    convert_list = divide_points_list_np.reshape(a, b)
    return convert_list

##############################################################################
class Individual:
    def __init__(self, dim, bounds, id):
        self.features = [random.uniform(bounds[x][0], bounds[x][1]) for x in range(dim)]
        self.ofv = 0
        self.id = id

    def result(self, x, dim):
        self.result =0.
        for i in range(dim):
            self.result += x[i] ** 2

        return self.result
    def __repr__(self):
        return str(self.__dict__)


class SHADE:

    def __init__(self, dim, maxFEs, bounds, H, NP, minPopSize):
        self.dim = dim
        self.maxFEs = maxFEs
        self.NP = NP
        self.F = None
        self.CR = None
        self.P = None
        self.eps = 10 ** (-5)
        self.Aext = None
        self.M_F = None
        self.M_CR = None
        self.S_F = None
        self.S_CR = None
        self.H = H
        self.Asize = None
        self.M_Fhistory = None
        self.M_CRhistory = None
        self.minPopSize = minPopSize
        self.maxPopSize = NP

    def getRandomInd(self, array, toRemove):
        popCopy = array[:]
        for i in toRemove:
            popCopy.remove(i)
        return random.choice(popCopy)

    def pickBests(self, size, id):
        popCopy = self.P[:]
        popCopy.remove(id)
        return sorted(popCopy, key=lambda ind: ind.ofv)[:size]

    def euclid(self, u, v):
        sum = 0
        for i in range(len(u)):
            sum += (u[i] - v[i])**2
        return sum**(1/2)

    def resizeAext(self):
        copy = sorted(self.Aext[:], key=lambda ind: ind.ofv)
        self.Aext = copy[:self.NP]

    def resize(self, array, size):
        copy = sorted(array[:], key=lambda ind: ind.ofv)
        return copy[:size]


    def mutation(self, x, pbest, xr1, xr2, F):
        v = list(range(self.dim))

        for i in range(self.dim):
            v[i] = x[i] + F * (pbest[i] - x[i]) + F * (xr1[i] - xr2[i])

        return v

    def crossover(self, original, v, CR):
        u = original[:]

        j = random.randint(0, self.dim)

        for i in range(self.dim):
            if (random.uniform(0, 1) <= CR) or (i == j):
                u[i] = v[i]

        return u

    def bound_constrain(self, original, u):

        for i in range(self.dim):
            if u[i] < bounds[i][0]:
                u[i] = (bounds[i][0] + original[i]) / 2
            elif u[i] > bounds[i][1]:
                u[i] = (bounds[i][1] + original[i]) / 2

        return u

    def run(self):

        #initialization
        G = 0
        self.Aext = []
        self.M_F = list(range(self.H))
        self.M_CR = list(range(self.H))
        best = None
        fes = 0

        k = 0
        pMin = 2/self.NP

        for i in range(0, self.H):
            self.M_F[i] = 0.5
            self.M_CR[i] = 0.5

        #population initialization
        id = 0
        self.P = [Individual(self.dim, bounds, id) for x in range(self.NP)]
        for ind in self.P:
            ind.ofv = object_fun(ind.features)
            ind.id = id
            id += 1
            fes += 1
            if best == None or ind.ofv <= best.ofv:
                best = ind

        #maxfes exhaustion
        while fes < self.maxFEs:
            G += 1
            newPop = []
            self.S_CR = []
            self.S_F = []
            wS = []

            #generation iterator
            for i in range(self.NP):
                original = self.P[i]

                r = random.randint(0, self.H -1)
                Fg = cauchy.rvs(loc=self.M_F[r], scale=0.1, size=1)[0]
                while(Fg <= 0):
                    Fg = cauchy.rvs(loc=self.M_CR[r], scale=0.1, size=1)[0]
                if(Fg > 1):
                    Fg = 1

                #CRg = cauchy.rvs(loc=self.M_CR[r], scale=0.1, size=1)[0]
                CRg = np.random.normal(self.M_CR[r], 0.1, 1)[0]
                if(CRg > 1):
                    CRg = 1
                if(CRg < 0):
                    CRg = 0

                Psize = round(random.uniform(pMin, 0.2) * self.NP)
                if(Psize < 2):
                    Psize = 2

                pBestArray = self.pickBests(Psize, original)

                #parent selection
                pbestInd = random.choice(pBestArray)

                xr1 = self.getRandomInd(self.P, [original, pbestInd])
                xr2 = self.getRandomInd(list(set().union(self.P, self.Aext)), [original, pbestInd, xr1])

                #mutation
                v = self.mutation(original.features, pbestInd.features, xr1.features, xr2.features, Fg)

                #crossover
                u = self.crossover(original.features, v, CRg)

                #bound constraining
                u = self.bound_constrain(original.features, u)

                #evaluation
                newInd = Individual(self.dim, bounds, original.id)
                newInd.features = u
                newInd.ofv = object_fun(u)
                fes += 1

                #selection step
                if newInd.ofv <= original.ofv:
                    newPop.append(newInd)
                    if newInd.ofv <= best.ofv:
                        best = newInd
                    self.S_F.append(Fg)
                    self.S_CR.append(CRg)
                    self.Aext.append(original)
                    wS.append(self.euclid(original.features, newInd.features))
                else :
                    newPop.append(original)

                if fes >= self.maxFEs:
                    return best

                if len(self.Aext) > self.NP:
                    self.resizeAext()

            self.P = newPop

            if len(self.S_F) > 0:
                wSsum = 0
                for i in wS:
                    wSsum += i

                meanS_F1 = 0
                meanS_F2 = 0
                meanS_CR1 = 0
                meanS_CR2 = 0

                for s in range(len(self.S_F)):
                    meanS_F1 += (wS[s] / wSsum) * self.S_F[s] * self.S_F[s]
                    meanS_F2 += (wS[s] / wSsum) * self.S_F[s]
                    meanS_CR1 += (wS[s] / wSsum) * self.S_CR[s] * self.S_CR[s]
                    meanS_CR2 += (wS[s] / wSsum) * self.S_CR[s]

                self.M_F[k] = (meanS_F1 / meanS_F2)
                if meanS_CR2 != 0:
                    self.M_CR[k] = (meanS_CR1 / meanS_CR2)
                else:
                    self.M_CR[k] = 0

                k += 1
                if k >= self.H:
                    k = 0

            self.NP = round(self.maxPopSize - (fes/self.maxFEs) * (self.maxPopSize - self.minPopSize))
            self.P = self.resize(self.P, self.NP)
            self.resizeAext()
        print(fes)
        return best
##########################################
print("TRY DIRECT")
direct_time_sta = time.perf_counter()
iter = 50000
num = int(args[4])
D = int(args[3])
bands = np.zeros((D, 2))
for i in range(D):
    bands[i] = [0, 1]

bands_list = DIRECT(object_fun, bands, iter, num)
if(args[1] == "rosenblock"):
    bands_list = bands_list * 10 - 5
else:
    bands_list = bands_list * 10.24 - 5.12

print(bands_list.shape)

print("DIRECT OK")
direct_time_end = time.perf_counter()
direct_time = direct_time_end - direct_time_sta
print('{}[s]'.format(direct_time))

low_time_start = time.perf_counter()
shade_time_start = time.perf_counter()
if(args[2] == "umap"):
    print("TRY UMAP")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.7)
    embedding = reducer.fit_transform(bands_list)
    print(len(embedding), embedding.shape)
    print(type(embedding))

    low_time_end = time.perf_counter()
    low_time = low_time_end - low_time_start
    print('{}[s]'.format(low_time))

    print("UMAP OK")
elif(args[2] == "tsne"):
    print("TRY t-SNE")
    tsne = TSNE(perplexity = 20, angle = 0.7)
    embedding = tsne.fit_transform(bands_list)
    print(len(embedding), embedding.shape)
    print(type(embedding))

    low_time_end = time.perf_counter()
    low_time = low_time_end - low_time_start
    print('{}[s]'.format(low_time))

dim = 2 #dimension size
NP = 500 #population size
maxFEs = 99500 #maximum number of objective function evaluations
F = 0.5
CR = 0.5
H = 500#archive size
minPopSize = 4
bounds = embedding


de = SHADE(dim, maxFEs, bounds, H, NP, minPopSize)
resp = de.run()
print(resp)
print(object_fun(resp.features))
shade_time_end = time.perf_counter()
shade_time = shade_time_end - shade_time_start
print('{}[s]'.format(shade_time))

f = open('result.txt', 'a')
if args[2] == "umap":
    f.write('D:{}, function={},num = {},maxFEs = {}, UMAP time:{}[s],min = {}\n'.format(D,args[1],num, maxFEs, low_time, object_fun(resp.features)))
if args[2] == "tsne":
    f.write('D:{}, function={},num = {},maxFEs = {}, t-SNE time:{}[s],min = {}\n'.format(D,args[1],num, maxFEs, low_time, object_fun(resp.features)))

f.close()
