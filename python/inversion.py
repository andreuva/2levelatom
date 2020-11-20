#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver  as fs
import random

a_sol = 1e-5                          # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-2                          # XCI/XLI
eps_sol = 1e-4                        # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 0                       # Depolirarization colisions (delta)
Hd_sol = 1                            # Hanle depolarization factor [1/5, 1]

print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)

random.seed(124)
a = random.uniform(1e-10,1)
r = random.uniform(1e-15,1)
eps = random.uniform(1e-4,1)
dep_col = random.uniform(0,10)
Hd = random.uniform(1/5, 1)
print("\nInitial parameters: ")
print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
print("Computing the profiles:")
I, Q = fs.solve_profiles(a, r, eps, dep_col, Hd)