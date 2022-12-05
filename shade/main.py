# Import necessary libraries
import time
import numpy as np
import SHADE as alg
from searchAlgorithm import searchAlgorithm
# Set global variables
#Double
OShift = M = y = z = x_bound = None
#variable
ini_flag=0
n_flag = func_flag = SS = None

#global variable
g_function_number = g_problem_size = g_max_num_evaluations = None
g_pop_size = g_arc_rate = g_memory_size = g_p_best_rate = None

# Main function
def main():
    # Number of runs
    num_runs = 51

    # Dimension size. Please select from 10, 30, 50, 100
    g_problem_size = 10

    # Available number of fitness evaluations
    g_max_num_evaluations = g_problem_size * 10000

    # Random seed is selected based on time according to competition rules
    np.random.randint(0, time.time())

    # SHADE parameters
    g_pop_size = 100
    g_memory_size = g_problem_size
    g_arc_rate = 2
    g_p_best_rate = 0.1

    for i in range(30):
        g_function_number = i + 1
        print("\n-------------------------------------------------------")
        print("Function = {}, Dimension size = {}\n".format(g_function_number, g_problem_size))

        bsf_fitness_array = np.zeros()
        mean_bsf_fitness = 0
        std_bsf_fitness = 0

        for j in range(num_runs):
            bsf_fitness_array[j] = alg.run()
            print("{}th run, error value = {}".format(j + 1, bsf_fitness_array[j]))

        for j in range(num_runs):
            mean_bsf_fitness += bsf_fitness_array[j]
        mean_bsf_fitness /= num_runs

        for j in range(num_runs):
            std_bsf_fitness += np.power(mean_bsf_fitness - bsf_fitness_array[j], 2.0)
        std_bsf_fitness /= num_runs
        std_bsf_fitness = np.sqrt(std_bsf_fitness)

        print("\nmean = {}, std = {}".format(mean_bsf_fitness, std_bsf_fitness))

if __name__ == "__main__":
    main()
