import numpy as np
from precomputeFx import *

all_nodes_180 = np.load('../../latticeFiles/all_nodes_latt180.npy',allow_pickle=True)

# precompute fx for all possible lattice locations for a given fault;
## fault_name: string for file save prefix to indicate fault
## fault: array with first element fault strike and second element fault dip
## arr_of_nodes: either all_nodes_180 or all_nodes_360
## num_nodes: typically len(arr_of_nodes)

test_fx_ = precompute_fx('test_fault_1',np.array([0.0,np.pi/2]),all_nodes_180,len(all_nodes_180))