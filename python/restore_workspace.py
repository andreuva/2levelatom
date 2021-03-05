#############################################################################
#       RESTORING ALL THE VARIABLES IN A WORKSPACE SAVED WITH SHELVE        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import shelve
import forward_solver_jkq  as fs
import forward_solver_py as sfs
import forward_solver_py_J as sfs_j

# Directory where the workspace_variables.out is located
directory = '../figures/130012438205_2000_1_1.0e+03_1.0e+06_1.0e+08_1.0e-02/'

my_shelf = shelve.open(directory + 'workspace_variables.out')
for key in my_shelf:
    try:
        globals()[key]=my_shelf[key]
    except:
        print(f'Failed to load {key}')
my_shelf.close()

""" 
Once run this with Ipython, you can play arround and look at all the variables.
Particulary interesting ones are the ones with "_evolution" since this ones
store info about each step of the inversion. 
"""