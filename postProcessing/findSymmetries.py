import numpy as np
import h5py

#--------Modify----------
fileName = '' #ex: 'rw_032522_0713_export.h5'
fname='test_'
#End---------------------


#--------Upload-Files----
imported_points=h5py.File('../stressPosteriors/'+fileName,'r')
f_=h5py.File('../latticeFiles/latt_inds_4dec_1801010.h5','r')
sphnode_index_dict=np.load('../latticeFiles/sph_index_dict_4dec_1801010.npy',allow_pickle=True).item()
rho_index_dict=np.load('../latticeFiles/rho_index_dict_4dec_1801010.npy',allow_pickle=True).item()

#--------Functions-------
def add_values_in_dict(sample_dict, key, list_of_values):
    """Append multiple values to a key in the given dictionary"""
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict

def find_symm_indices(points):
    trial_sym_inds=np.zeros((len(points),4))
    for k in range(points.shape[0]):
        point=np.round(points[k],decimals=4)
        phi_theta=str(point[0])+'_'+str(point[1])
        rho=str(point[2])
        Delta=str(point[3])
        R=str(point[4])
        trial_phi_theta_ind=f_['phi_theta'][phi_theta][:][0]
        trial_rho_ind=f_['rho'][rho][:][0]
        trial_Delta_ind=f_['Delta'][Delta][:][0]
        trial_R_ind=f_['R'][R][:][0]
        trial_ind=trial_phi_theta_ind+trial_rho_ind+trial_Delta_ind+trial_R_ind
        spsyminds=sphnode_index_dict[trial_phi_theta_ind]
        rsyminds=rho_index_dict[trial_rho_ind]
        trial_sym_inds[k]=[a + b for a, b in zip(spsyminds, rsyminds)]+trial_Delta_ind+trial_R_ind
    return trial_sym_inds#,check

def find_symm_for_indices(indices):
    trial_sym_inds=np.zeros((len(indices),4))
    for k in range(len(indices)):
        index=indices[k]
        rem_rDR = index%18000
        pt_sym=int(index-rem_rDR)
        rem_DR= rem_rDR%100
        r_sym=int(rem_rDR-rem_DR)
        rem_R = rem_DR%10
        D_sym=int(rem_DR-rem_R)
        R_sym=int(rem_R)
        spsyminds=sphnode_index_dict[pt_sym]
        rsyminds=rho_index_dict[r_sym]
        ptr_indsum = [a + b for a, b in zip(spsyminds, rsyminds)]
        trial_sym_inds[k]=[x+D_sym+R_sym for x in ptr_indsum]
    return trial_sym_inds

def find_point_from_index(index):
    rem_rDR = index%18000
    pt_sym=int(index-rem_rDR)
    rem_DR= rem_rDR%100
    r_sym=int(rem_rDR-rem_DR)
    rem_R = rem_DR%10
    D_sym=int(rem_DR-rem_R)
    R_sym=int(rem_R)
    return np.hstack((f_['ind_pt'][str(pt_sym)][:],f_['ind_r'][str(r_sym)][0],f_['ind_D'][str(D_sym)][0],f_['ind_R'][str(R_sym)][0])
)

#----------------------
#convert points to indices and find symmetric indices
mcmc_pts=imported_points['pts_and_avg_rake'][:,:-1]
mcmc_pts=np.round(mcmc_pts,decimals=4)
mcmc_symm_indices=find_symm_indices(mcmc_pts)
print('mcmc indices found')
#find unique indices and their counts
unqinds,unqcts=np.unique(mcmc_symm_indices,return_counts=True,axis=0)

#append indices of symmetric indices not originally visited by mcmc
notvisitedinds=[]
unqvisitedinds=unqinds[:,0]
for i in range(unqinds.shape[0]):
    for j in range(1,4):
        if unqinds[i,j] not in unqvisitedinds:
            notvisitedinds.append(unqinds[i,j])
        else:
            unqcts[np.where(unqinds[i,j]==unqvisitedinds)[0]]+=1

notvisitedunq,notvisitedcts=np.unique(notvisitedinds,return_counts=True)
inds_of_mcmc_and_unvisited=np.hstack((unqinds[:,0],notvisitedunq))
counts_of_mcmc_and_unvisited=np.hstack((unqcts,notvisitedcts))

np.save(fname+'counts_with_symmetries.npy',arr=counts_of_mcmc_and_unvisited)
np.save(fname+'unique_points_with_symmetries.npy',arr=inds_of_mcmc_and_unvisited)
