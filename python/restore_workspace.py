import numpy as np
import matplotlib.pyplot as plt
import shelve

directory = '../figures/130555883702_1500_0.1_100.0_2.10e-01/'


my_shelf = shelve.open(directory + 'variables.out')
for key in my_shelf:
    try:
        globals()[key]=my_shelf[key]
    except:
        print(f'Failed to load {key}')
my_shelf.close()