import forward_solver_py as fs
import numpy as np

errors = 0
bad_pars = []

for a in [1,1e-3, 1e-6, 1e-12]:
    for r in [10, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12]:
        for eps in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            for dep_col in [10, 1, 1e-2, 0]:
                for Hd in [1,0.5, 0.2]:
                    print(f" a = {a}\t r = {r}\t eps = {eps}\t delta = {dep_col}\t Hd = {Hd}\n")
                    I,Q = fs.solve_profiles( a, r, eps, dep_col, Hd)
                    if np.min(I) < 0:
                        errors = errors + 1
                        bad_pars.append([a,r,eps,dep_col,Hd])

bad_pars = np.array(bad_pars)
print(bad_pars)
np.savetxt('test_bad_params_2.csv', bad_pars)