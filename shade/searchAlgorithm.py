import numpy as np
import main as x
from typing import List, Tuple

class searchAlgorithm:

    def __init__(self, g_function_number, g_problem_size, g_max_num_evaluations, g_pop_size):
        # パラメータの初期化
        self.function_number = g_function_number
        self.problem_size = g_problem_size
        self.max_num_evaluations = g_max_num_evaluations
        self.pop_size = g_pop_size
        self.initializeFitnessFunctionParameters()

    def rand_double(self) -> float:
            return np.random.rand()

    def cauchy_g(self, mu: float, gamma: float) -> float:
        return mu + gamma * np.tan(np.pi * (self.rand_double() - 0.5))

    def gauss(self, mu: float, sigma: float) -> float:
        return mu + sigma * np.sqrt(-2.0 * np.log(self.rand_double())) * np.sin(2.0 * np.pi * self.rand_double())

    def sort_index_with_quicksort(self, array: np.ndarray, first: int, last: int, index: np.ndarray):
        x = array[(first + last) // 2]
        i = first
        j = last

        while True:
            while array[i] < x:
                i += 1
            while x < array[j]:
                j -= 1
            if i >= j:
                break

            array[i], array[j] = array[j], array[i]
            index[i], index[j] = index[j], index[i]

            i += 1
            j -= 1

        if first < (i - 1):
            self.sort_index_with_quicksort(array, first, i - 1, index)
        if (j + 1) < last:
            self.sort_index_with_quicksort(array, j + 1, last, index)

    def _initializeFitnessFunctionParameters(self):
        # epsilonは受け入れ可能なエラー値
        self.epsilon = pow(10.0, -8)
        self.max_region = 100.0
        self.min_region = -100.0
        self.optimum = self.function_number * 100

    def _evaluatePopulation(self, pop: List[List[float]], fitness: List[float]):
        for i in range(self.pop_size):
            #function(pop[i],  fitness[i], self.problem_size, 1, self.function_number)

    def _setBestSolution(self, pop: List[List[float]], fitness: List[float], bsf_solution: List[float], bsf_fitness: float):
        current_best_individual = 0

        for i in range(1, self.pop_size):
            if fitness[current_best_individual] > fitness[i]:
                current_best_individual = i

        bsf_fitness = fitness[current_best_individual]
        for i in range(self.problem_size):
            bsf_solution[i] = pop[current_best_individual][i]

    def _makeNewIndividual(self) -> List[float]:
        individual = [0.0] * self.problem_size

        for i in range(self.problem_size):
            individual[i] = ((self.max_region - self.min_region) * np.random.random()) + self.min_region

        return individual

    def _modifySolutionWithParentMedium(self, child: List[float], parent: List[float]):
        for j in range(self.problem_size):
            if child[j] < self.min_region:
                child[j]= (self.min_region + parent[j]) / 2.0
            elif child[j] > self.max_region:
                child[j]= (self.max_region + parent[j]) / 2.0
