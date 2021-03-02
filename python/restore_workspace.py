import numpy as np
import matplotlib.pyplot as plt
import shelve
import forward_solver_jkq  as fs
import forward_solver_py as sfs
import forward_solver_py_J as sfs_j

directory = '../figures/115702436353_3000_0.1_100.0_2.00e-01/'


my_shelf = shelve.open(directory + 'variables.out')
for key in my_shelf:
    try:
        globals()[key]=my_shelf[key]
    except:
        print(f'Failed to load {key}')
my_shelf.close()

print('\nFound Parameters - Solution parameters:')
print('Initial   &  %1.2e &  %1.2e &  %1.2e &  %1.2e &  %1.2e \\ \hline' % (a_initial, r_initial, eps_initial, dep_col_initial, Hd_initial))
print('Inversion &  %1.2e &  %1.2e &  %1.2e &  %1.2e &  %1.2e \\ \hline' % (a_res, r_res, eps_res, dep_col_res, Hd_res))
print('Solution  &  %1.2e &  %1.2e &  %1.2e &  %1.2e &  %1.2e \\ \hline' % (a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol))