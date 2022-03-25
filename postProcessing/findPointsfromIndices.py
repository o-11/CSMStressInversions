import numpy as np
import h5py

#Modify-------------------------
name='test_' #index file prefix; and save name
#End----------------------------

#---------------Load-files------
indices=np.load(name+'unique_points_with_symmetries.npy',allow_pickle=True)
f_=h5py.File('../latticeFiles/latt_inds_4dec_1801010.h5','r')

#---------------Functions--------
def find_point_from_index(index):
    rem_rDR = index%18000
    pt_sym=int(index-rem_rDR)
    rem_DR= rem_rDR%100
    r_sym=int(rem_rDR-rem_DR)
    R_sym = int(rem_DR%10)
    D_sym=int(rem_DR-R_sym)
    return np.hstack((f_['ind_pt'][str(pt_sym)][:],f_['ind_r'][str(r_sym)][0],f_['ind_D'][str(D_sym)][0],f_['ind_R'][str(R_sym)][0]))

#-------------
points=[None]
points=[find_point_from_index(ind_) for ind_ in indices]
points_arr=np.array(points)
print('saving points...')
np.save(name+'idxconverted_points.npy',points_arr,allow_pickle=True)
f_.close()
print('complete')
