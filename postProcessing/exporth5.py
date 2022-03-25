import numpy as np
import h5py

#Modify------------------
fileNamePrefix = '../stressPosteriors/rw_032522_0713_'
numFaults = 2
numSteps = 5*10**6 #5000000
numFiles = 250 #250
#End----------------------

def export_to_single_h5(file_prefix,num_faults,num_files=250,len_file=20000,num_steps=5000000):
    for j in range(1,num_files+1):
        name = str(file_prefix)+str(j)+'.hdf5'
        f=h5py.File(str(name),'r')
        new_file = str(file_prefix)+'all.h5'
        with h5py.File(str(new_file),'a') as h5a:
            for group in f.keys():
                print(group)
                for ds in f[group].keys():
                    ds_arr = f[group][ds][:]
                    print(ds,':',ds_arr.dtype,ds_arr.shape)
                    h5a.create_dataset(group+'/'+ds, data=ds_arr)
        print('done')
        f.close()
    f=h5py.File(str(new_file),'r')
    export_file = str(file_prefix)+'export.h5'
    with h5py.File(str(export_file),'a') as h5w:
        h5w.create_dataset('pts_and_avg_rake', shape=(num_steps,6))
        for i in range(1,num_files+1):
            print(i)
            ds_arr = f['pts_and_avg_rake'][str(i)]
            print(ds_arr.dtype, ds_arr.shape)
            h5w['pts_and_avg_rake'][len_file*(int(i)-1):len_file*(int(i))]=ds_arr[:len_file]
        h5w.create_dataset('all_predicted_rakes', shape=(num_steps,num_faults))
        for i in range(1,num_files+1):
            print(i)
            ds_arr = f['all_predicted_rakes'][str(i)]
            print (ds_arr.dtype, ds_arr.shape)
            h5w['all_predicted_rakes'][len_file*(int(i)-1):len_file*(int(i))]=ds_arr[:len_file]
    f.close()
    return print(f'{file_prefix} files exported to {file_prefix}export.h5 successfully')

export_to_single_h5(fileNamePrefix,numFaults,num_files=numFiles,len_file=numSteps//numFiles,num_steps=numSteps)
