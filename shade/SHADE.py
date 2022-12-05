import numpy as np
from collections import deque

import main as x
import searchAlgorithm as s


class SHADE:
    # initialize population
    def run(self):
        pop = []
        fitness = [0 for i in range(x.pop_size)]
        children = []
        children_fitness = [0 for i in range(x.pop_size)]


        for i in range(s.pop_size):
            pop.append(s.makeNewIndividual())
            children.append(np.zeros(s.problem_size, dtype=np.int))
            # evaluate the initial population's fitness values
        s.evaluatePopulation(pop, fitness)

        bsf_solution = np.zeros(s.problem_size, dtype=np.int)
        bsf_fitness = None
        nfes = 0

        if (fitness[0] - s.optimum) < s.epsiron:
            fitness[0] = s.optimum
        bsf_fitness = fitness[0]

        for j in range(s.problem_size):
            bsf_solution[j] = pop[0][j]

        # iterate over the population
        for i in range(s.pop_size):
            nfes += 1

            if (fitness[i] - s.optimum) < s.epsilon:
                fitness[i] = s.optimum

            if fitness[i] < bsf_fitness:
                bsf_fitness = fitness[i]
                for j in range(s.problem_size):
                    bsf_solution[j] = pop[i][j]

            if nfes >= s.max_num_evaluations:
                break

        # evaluate the initial population's fitness values
        s.evaluatePopulation(pop, fitness)

        bsf_solution = np.zeros(s.problem_size)
        nfes = 0

        if (fitness[0] - s.optimum) < s.epsilon:
            fitness[0] = s.optimum
        bsf_fitness = fitness[0]

        for j in range(s.problem_size):
            bsf_solution[j] = pop[0][j]

        # iterate over the population
        for i in range(s.pop_size):
            nfes += 1

            if (fitness[i] - s.optimum) < s.epsilon:
                fitness[i] = s.optimum

            if fitness[i] < bsf_fitness:
                bsf_fitness = fitness[i]
                for j in range(s.problem_size):
                    bsf_solution[j] = pop[i][j]

            if nfes >= s.max_num_evaluations:
                break

        arc_ind_count = 0
        random_selected_arc_ind = 0
        archive = []
        for i in range(arc_size):
            archive.append(s.problem_size)

        num_success_params = 0
        success_sf = []
        success_cr = []
        dif_fitness = []

        memory_sf = deque([0.5] * memory_size, maxlen=memory_size)
        memory_cr = deque([0.5] * memory_size, maxlen=memory_size)

        temp_sum_sf = 0
        temp_sum_cr = 0
        sum = 0
        weight = 0

        memory_pos = 0

        mu_sf = 0
        mu_cr = 0
        random_selected_period = 0
        pop_sf = np.zeros(s.problem_size)
        pop_cr = np.zeros(s.problem_size)

        p_best_ind = 0
        p_num = round(s.pop_size *  p_best_rate)
        sorted_array = np.zeros(s.problem_size)
        temp_fit = np.zeros(s.problem_size)

        #main loop
        while nfes < s.max_num_evaluations:
            for i in range(s.pop_size):
                sorted_array[i] = i
            for i in range(s.pop_size):
                temp_fit[i] = fitness[i]
            s.sort_index_with_quicksort(temp_fit, 0, s.pop_size - 1, sorted_array)

            for target in range(s.pop_size):
                random_selected_period = np.random.randint(0, memory_size - 1)
                mu_sf = memory_sf[random_selected_period]
                mu_cr = memory_cr[random_selected_period]

                if mu_cr == -1:
                    pop_cr[target] = 0
                else:
                    pop_cr[target] = s.gauss(mu_cr, 0.1)
                    if pop_cr[target] > 1:
                        pop_cr[target] = 1
                    elif pop_cr[target] < 0:
                        pop_cr[target] = 0

                while True:
                    pop_sf[target] = s.cauchy_g(mu_sf, 0.1)
                    if pop_sf[target] > 0:
                        break
                if pop_sf[target] > 1:
                    pop_sf[target] = 1

                p_best_ind = sorted_array[np.random.randint(0, p_num - 1)]
                s.operate_current_to_pbest1_bin_with_archive(pop, children[target], target, p_best_ind, pop_sf[target], pop_cr[target], archive, arc_ind_count)

            s.evaluatePopulation(children, children_fitness)

            for i in range(s.pop_size):
                nfes += 1

                if (children_fitness[i] - s.optimum) < s.epsilon:
                    children_fitness[i] = s.optimum
                if children_fitness[i] < bsf_fitness:
                    bsf_fitness = children_fitness[i]
                    for j in range(s.problem_size):
                        bsf_solution[j] = children[i][j]
                if nfes >= s.max_num_evaluations:
                    break

            for i in range(s.pop_size):
                if children_fitness[i] == fitness[i]:
                    fitness[i] = children_fitness[i]
                    for j in range(s.problem_size):
                        pop[i][j] = children[i][j]
                elif children_fitness[i] < fitness[i]:
                    if arc_size > 1:
                        if arc_ind_count < arc_size:
                            for j in range(s.problem_size):
                                archive[arc_ind_count][j] = pop[i][j]
                            arc_ind_count += 1
                        else:
                            random_selected_arc_ind = np.random.randint(0, arc_size - 1)
                            for j in range(s.problem_size):
                                archive[random_selected_arc_ind][j] = pop[i][j]
                    dif_fitness.append(abs(fitness[i] - children_fitness[i]))
                    fitness[i] = children_fitness[i]
                    for j in range(s.problem_size):
                        pop[i][j] = children[i][j]
                    success_sf.append(pop_sf[i])
                    success_cr.append(pop_cr[i])

            num_success_params = len(success_sf)

            if num_success_params > 0:
                memory_sf[memory_pos] = 0
            memory_cr[memory_pos] = 0
            temp_sum_sf = 0
            temp_sum_cr = 0
            sum = 0

            for i in range(num_success_params):
                sum += dif_fitness[i]

            for i in range(num_success_params):
                weight = dif_fitness[i] / sum
                memory_sf[memory_pos] += weight * success_sf[i] * success_sf[i]
                temp_sum_sf += weight * success_sf[i]
                memory_cr[memory_pos] += weight * success_cr[i] * success_cr[i]
                temp_sum_cr += weight * success_cr[i]

            memory_sf[memory_pos] /= temp_sum_sf

            if temp_sum_cr == 0 or memory_cr[memory_pos] == -1:
                memory_cr[memory_pos] = -1
            else:
                memory_cr[memory_pos] /= temp_sum_cr

            memory_pos += 1
            if memory_pos >= memory_size:
                memory_pos = 0

            success_sf.clear()
            success_cr.clear()
            dif_fitness.clear()

            return bsf_fitness

    def operate_current_to_pbest1_bin_with_archive(self, pop, child, target, p_best_individual, scaling_factor, cross_rate, archive, arc_ind_count):
        # Create a NumPy array with random integer values.
        r1, r2 = np.random.randint(s.pop_size, size=2)

        # Check if the random integer values are valid.
        while r1 == target:
            r1 = np.random.randint(s.pop_size)
        while (r2 == target) or (r2 == r1):
            r2 = np.random.randint(s.pop_size + arc_ind_count)

        # Create a NumPy array with a random integer value.
        random_variable = np.random.randint(s.problem_size)

        # Calculate the new values of the child solution.
        if r2 >= s.pop_size:
            r2 -= s.pop_size
            # Calculate the new values of the child solution using NumPy arrays.
            mask = (np.random.random(s.problem_size) < cross_rate) | (random_variable == np.arange(searchAlgorithm.problem_size))
            child[mask] = pop[target][mask] + scaling_factor * (pop[p_best_individual][mask] - pop[target][mask]) + scaling_factor * (pop[r1][mask] - archive[r2][mask])
            child[~mask] = pop[target][~mask]
        else:
            # Calculate the new values of the child solution using NumPy arrays.
            mask = (np.random.random(s.problem_size) < cross_rate) | (random_variable == np.arange(searchAlgorithm.problem_size))
            child[mask] = pop[target][mask] + scaling_factor * (pop[p_best_individual][mask] - pop[target][mask]) + scaling_factor * (pop[r1][mask] - pop[r2][mask])
            child[~mask] = pop[target][~mask]

        # If the mutant vector violates bounds, the bound handling method is applied
        s.modify_solution_with_parent_medium(child, pop[target])

    def set_SHADE_parameters(self):
        global arc_rate, arc_size, p_best_rate, memory_size

        arc_rate = s.g_arc_rate
        arc_size = round(s.pop_size * arc_rate)
        p_best_rate = s.g_p_best_rate
        memory_size = s.g_memory_size


