import copy
import numpy as np
import cvxpy as cp


def ks(values, weights, cap):
    sol = np.zeros(len(values))
    unsorted_sol = np.zeros(len(values))
    p = values / weights
    i = np.argsort(-1*p)    # descending order
    sorted_p = p[i]
    sorted_w = weights[i]
    sorted_values = values[i]

    cum_weight = np.cumsum(sorted_w)
    break_index = np.where(cum_weight > cap)[0][0]
    sol[0:break_index] = 1
    sol[break_index] = (cap - cum_weight[break_index - 1]) / sorted_w[break_index]

    unsorted_sol[i] = sol

    ## figuring out the max:
    max_sol = copy.copy(unsorted_sol)
    frac_index = np.where(np.logical_and(max_sol < 1, max_sol>0))[0]
    if np.size(frac_index) >= 1:
        frac_index = frac_index[0]

        non_frac_value = max_sol @ values
        non_frac_value = non_frac_value - (max_sol[frac_index] * values[frac_index])
        frac_value = values[frac_index]

        # non_frac_value = sol @ values
        # non_frac_value = non_frac_value - values[frac_index]
        # frac_value = sol[frac_index]*values[frac_index]

        if frac_value <= non_frac_value:
            max_sol[frac_index] = 0
        else:
            max_sol = np.zeros(len(unsorted_sol))
            max_sol[frac_index] = 1     # for the feasibility guarantee, no item shall be bigger than the capacity itself

    ## figuring out the randomized
    rand_sol_cand_1 = copy.copy(unsorted_sol)
    rand_sol_cand_1[frac_index] = 0
    rand_sol_cand_2 = np.zeros(len(unsorted_sol))
    rand_sol_cand_2[frac_index] = 1

    if np.random.random() <= 0.5:
        random_sol = rand_sol_cand_1
    else:
        random_sol = rand_sol_cand_2

    # unsorted_sol is almost integral, max_sol is the max between integral and non integral, random sol is a sample
    return unsorted_sol, max_sol, random_sol

# dim = 5
# values = np.random.randint(1, 100, dim)
# weights = np.random.randint(1, 6, dim)
# cap = 6
# my_sol = np.round(ks(values, weights, cap), 4)
# print(my_sol)
# print("values: ", values)
# print("weights: ", weights)

##################################################################
##################################################################

# dim = 1000
# y = cp.Variable(dim)
# values_param = cp.Parameter(dim, nonneg=True)
# for i in range (1000):
#     values = np.random.randint(1, 11, dim)
#     weights = np.random.randint(1, 11, dim)
#     cap = 100
#
#     standard_constraints = []
#     standard_constraints.extend([y >= 0, y <= 1])
#     standard_constraints.extend([weights@y == cap])
#     values_param.value = values
#     objective = cp.Maximize(values_param@y)
#     prob = cp.Problem(objective, standard_constraints)
#     result = prob.solve(warm_start=True, ignore_dpp=True)
#
#     cpsol = np.round(y.value, 4)
#     my_sol = ks(values, weights, cap)[0]
#     my_sol = np.round(my_sol, 4)
#
#     # print("Constraint satisfaction:")
#     # print ("My sol: ", weights@my_sol)
#     # print("CP: ", weights @ cpsol)
#     # print("Function value:")
#     # print ("My sol: ", values@my_sol)
#     # print("CP: ", prob.value)
#     # print("====================")
#
#     if (values @ my_sol - prob.value ) > 0.01 :
#         print("!!")
#         print("Function value:")
#         print ("My sol: ", values@my_sol)
#         print("CP: ", prob.value)
#
#



