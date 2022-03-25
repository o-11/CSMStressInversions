import numpy as np
import datetime
import os
import h5py

# support functions that stress_functions rely upon:
from support_stress_functions import *


def start_loc(fault_data, lattice, dictionaries, given="empty", mle=True, lattice_nodes=[180, 10, 10]):
    """
    starting location for mcmc_v01 parameter starting_loc

    option to provide a start point: start_loc(*fault_data,*lattice,given=[start_point],mle=False)
    option to calculate most likely estimate (p-axis): start_loc(*fault_data,*lattice)
    option to calculate random start point: start_loc(*fault_data,*lattice,mle=False)

    """

    # unpack nodes and neighbors for ϕ,θ, and lattices for \rho, \Delta and R from h5 file
    sph_nodes, sph_neighbors, ρ_latt, Δ_latt, R_latt = lattice['sph_nodes'], lattice['sph_neighbors'][...], lattice['ρ_latt'], lattice['Δ_latt'], lattice['R_latt']
    fault_geometry, seg_potency, num_faults = fault_data

    if given != "empty":

        starting_node_locs, start_ind = given

    else:

        if mle is True:
            p_ax = np.zeros((num_faults, 2))
            for f in range(num_faults):
                mt = fault_plane_to_moment_tensor(fault_geometry[f, 0], fault_geometry[f, 1], fault_geometry[f, 2])
                p_ax[f] = moment_tensor_to_pnt(mt)[0]
            ###find average (angular) p-axis weighted by potency of faults
            avg_p_ax_trend = wtd_ang_avg(p_ax[:, 0], seg_potency)
            avg_p_ax_plunge = wtd_ang_avg(p_ax[:, 1], seg_potency)
            phi_theta = np.array([avg_p_ax_trend, avg_p_ax_plunge])

        else:
            phi_theta = np.array([np.random.rand(-np.pi, np.pi), np.random.rand(-np.pi / 2, np.pi / 2)])

        l = find_nearest_point(sph_nodes, phi_theta)[0]
        ρ_ind = np.random.randint(0, lattice_nodes[0])  # find random start location in \rho lattice
        Δ_ind = np.random.randint(0, lattice_nodes[1])  # '' '' '' '' \delta lattice
        R_ind = np.random.randint(0, lattice_nodes[2])  # '' '' '' '' \varsigma lattice

        starting_node_locs = np.hstack((sph_nodes[l], ρ_latt[ρ_ind], Δ_latt[Δ_ind], R_latt[R_ind]))
        start_ind = [l, ρ_ind, Δ_ind, R_ind]

    return starting_node_locs, start_ind


def mcmc_v01(rep_num, num_steps, lattice, slip_model, uncertainties, dictionaries, starting_loc='empty',
             lattice_nodes=[180, 10, 10], save_to_file='file_name'):
    """
    bayesian monte carlo markov chain for slip model inversions

    Inputs:
        num_steps: suggested number of steps for random walk over most compressive stress
        lattice: spherical lattice nodes and neighbors, and \rho, \delta and \varsigma lattices;
        slip_model: sm_data (slip model data configured from original authors' dataset) and fault_data (computed with weighted_fault_avg(sm_data))
        uncertainties: all_fx_for_sm (precomputed Jacobians for given fault strike and dip measurements for every possible model on current lattice with current lattice spacings; used for error propagation) and rake_unc (uncertainty on rake; precomputed for each slip model)
        dictionaries: key_dictionary and index_dictionary, for querying all_fx_for_sm
        lattice_nodes: number of \rho, \delta, and \varisgma (R) nodes;

    Outputs:
        [points[-1], [l, ρ_ind, Δ_ind, R_ind]]: last lattice location visited and its associated indices
        pts_and_avg_rake: all lattice locations visited and the average rake computed across the segments
        all_predicted_rakes: all rakes predicted for the segments for all lattice locations (ie stress models)
        data_var: log of max and min uncertainties; useful for debugging
        LR_: log of likelihood ratios; can help for debugging to ensure this is nonzero and changing

    """

    # unpack nodes and neighbors for ϕ,θ, and lattices for \rho, \Delta and R
    sph_nodes, sph_neighbors, ρ_latt, Δ_latt, R_latt = lattice['sph_nodes'], lattice['sph_neighbors'][...], lattice[
        'ρ_latt'], lattice['Δ_latt'], lattice['R_latt']
    # unpack slip model data and fault data
    sm_data, fault_data = slip_model
    # unpack fault geometry, segment potency and number of fault segments in slip model
    fault_geometry, seg_potency, num_faults = fault_data
    # weighted angular average rake for each fault segment
    obs_λ = np.array([fault_geometry[:, 2]]).T
    # dictionaries required for querying error propagated uncertainty
    key_dictionary, index_dictionary = dictionaries
    # unpack precomputed jacobians, uncertainty on rake and uncertainty on fault geometry (strike, dip)
    all_fx_for_sm, std_λ_, std_geometry = uncertainties

    # threshold for coulomb stress friction minimum; numerically required for predict_rake() stability, however, this is rarely a problem
    friction_threshold = 0.002

    # get starting location on lattice
    start_pt, start_ind = start_loc(fault_data, lattice, dictionaries, given=starting_loc, mle=True,
                                    lattice_nodes=[180, 10, 10])
    l, ρ_ind, Δ_ind, R_ind = start_ind
    print('starting location: ', start_pt * [180 / np.pi, 180 / np.pi, 180 / np.pi, 1, 1])

    # calculate prediction of rake for starting stress model
    start_rake, start_friction = predict_rake(fault_geometry, start_pt, num_faults)
    # confirm that shear stress for starting stress model is non-zero
    friction_bool = 1.0
    for f in range(num_faults):
        if np.abs(start_friction[f]) < friction_threshold:
            friction_bool = 0.0
            continue
    # get associated jacobian for stress model for each segment in slip model
    model_fx = get_fx_for_sm(all_fx_for_sm, num_faults, key_dictionary, index_dictionary, start_pt)
    # compute error propagated uncertainty from jacobians and uncertainty on strike and dip for each segment
    model_σ_P = σ_P_(model_fx, std_geometry)
    # compute relative likelihood for trial stress model for all segments in slip model given observed rake, predicted
    #   rake, uncertainty on rake and error propagated uncertainty from fault geometry:
    LL_last_pt, σ_tot = Gauss_Ll_prop(num_faults, seg_potency, obs_λ, start_rake, std_λ_, model_σ_P)

    # begin storing min and max uncertainty values computed through out random walk; not needed but can inspect in output logs
    σ_min = np.hstack((np.array([σ_tot]).T, np.array([np.repeat(0, len(σ_tot))]).T))
    σ_max = np.hstack((np.array([σ_tot]).T, np.array([np.repeat(0, len(σ_tot))]).T))
    # begin storing all predicted rakes, average rakes, points visited, and likelihood ratio
    all_predicted_rakes = np.zeros((num_steps + 1, num_faults))
    avg_predicted_rake = np.zeros((num_steps + 1, 1))
    points = np.zeros((num_steps + 1, 5))
    LR_ = np.zeros((num_steps + 1, 1))
    all_predicted_rakes[0] = (start_rake).T
    avg_predicted_rake[0] = wtd_ang_avg(start_rake, seg_potency)
    points[0] = start_pt

    k = 0
    while (k < num_steps):

        # select next stress model randomly from neighboring lattice nodes
        i = np.random.randint(0, len(sph_neighbors[l]))
        ρ_ind_ = np.random.randint(-1, 2) + ρ_ind
        if ρ_ind_ > (lattice_nodes[0] - 1):
            ρ_ind_ = 0
        elif ρ_ind_ < 0:
            ρ_ind_ = lattice_nodes[0] - 1
        Δ_ind_ = np.random.randint(-1, 2) + Δ_ind
        R_ind_ = np.random.randint(-1, 2) + R_ind
        if Δ_ind_ > (lattice_nodes[1] - 1) or Δ_ind_ < 0 or R_ind_ > (lattice_nodes[2] - 1) or R_ind_ < 0:
            points[k + 1] = points[k]
            avg_predicted_rake[k + 1] = avg_predicted_rake[k]
            all_predicted_rakes[k + 1] = all_predicted_rakes[k]
            continue

        sph_ind = int(sph_neighbors[l][i])
        trial_sph = sph_nodes[sph_ind]
        trial_pt = np.hstack((trial_sph, ρ_latt[ρ_ind_], Δ_latt[Δ_ind_], R_latt[R_ind_]))

        trial_rake, trial_friction = predict_rake(fault_geometry, trial_pt, num_faults)
        friction_bool = 1.0
        for f in range(num_faults):
            if np.abs(trial_friction[f]) < friction_threshold:
                friction_bool = 0.0
        model_fx = get_fx_for_sm(all_fx_for_sm, num_faults, key_dictionary, index_dictionary, trial_pt)
        model_σ_P = σ_P_(model_fx, std_geometry)
        LL_trial_pt, σ_tot = Gauss_Ll_prop(num_faults, seg_potency, obs_λ, trial_rake, std_λ_, model_σ_P)

        for f in range(num_faults):
            if σ_tot[f] > σ_max[f, 0]:
                σ_max[f] = np.hstack((σ_tot[f], k + 1))
            elif σ_tot[f] < σ_min[f, 0]:
                σ_min[f] = np.hstack((σ_tot[f], k + 1))

        # compute likelihood ratio
        if LL_trial_pt - LL_last_pt > 709:
            LR = friction_bool * np.exp(709)  # avoids np.exp overflow error; exp(709) is approx E+32, upper limit for float64
        else:
            LR = friction_bool * np.exp(LL_trial_pt - LL_last_pt)

        if LR >= np.random.uniform(0.0, 1.0):
            points[k + 1] = trial_pt
            l = sph_ind
            ρ_ind = ρ_ind_
            Δ_ind = Δ_ind_
            R_ind = R_ind_
            LL_last_pt = LL_trial_pt

            avg_predicted_rake[k + 1] = wtd_ang_avg(trial_rake, seg_potency)
            all_predicted_rakes[k + 1] = (trial_rake).T

        else:
            points[k + 1] = points[k]
            avg_predicted_rake[k + 1] = avg_predicted_rake[k]
            all_predicted_rakes[k + 1] = all_predicted_rakes[k]

        LR_[k + 1] = LR
        k += 1
    data_var = [σ_min, σ_max]
    pts_and_avg_rake = np.hstack((points, avg_predicted_rake))

    return [points[-1], [l, ρ_ind, Δ_ind, R_ind]], pts_and_avg_rake, all_predicted_rakes, data_var, LR_


def mcmc_implementation(num_steps, lattice, initial_starting_loc, slip_model, uncertainties, dictionaries,
                        save_to_file='default', lattice_nodes=[180, 10, 10]):
    """
    runs mcmc v01 multiple times, restarts at last point
    allows code checkpoints and timer outputs for each iteration
    """

    rep_num = 1

    # create initial hdf5 files for data_sets to be stored:
    if save_to_file == 'default':
        file_name = 'rw_' + str(datetime.datetime.today().strftime("%m%d%y_%H%M"))
        LRfile_name = 'lr_' + str(datetime.datetime.today().strftime("%m%d%y_%H%M"))
    else:
        file_name = save_to_file
        LRfile_name = 'lr_' + save_to_file
    f = h5py.File(str(file_name) + '_' + str(rep_num) + '.hdf5', 'w')

    grp_pts_and_avg_rake = f.create_group("pts_and_avg_rake")
    grp_all_pred_rake = f.create_group("all_predicted_rakes")
    grp_data_var = f.create_group("data_var")

    # determine number of checkpoints based upon length of RW
    if num_steps < 20000:
        indiv_steps = num_steps
    else:
        indiv_steps = 20000
    reps = num_steps // indiv_steps

    # for rep 1:
    time_ = datetime.datetime.today().strftime("%H:%M:%S")
    print(f'{rep_num} of {reps} began at {time_} \n')
    runtimelog = open('runtimelog' + str(file_name) + '.txt', 'w')
    runtimelog.write('start: ' + str(time_))
    runtimelog.close()

    last_pt, pts_and_avg_rake, all_predicted_rakes, data_var, lr_ = mcmc_v01(rep_num, indiv_steps, lattice, slip_model,
                                                                             uncertainties, dictionaries,
                                                                             starting_loc=initial_starting_loc,
                                                                             lattice_nodes=[180, 10, 10],
                                                                             save_to_file=file_name)

    LR_file = open(str(LRfile_name) + '.txt', 'w')
    LR_file.write(str(lr_[:]) + '\n')
    LR_file.close()

    time_ = datetime.datetime.today().strftime("%H:%M:%S")
    runtimelog = open('runtimelog' + str(file_name) + '.txt', 'a')
    runtimelog.write(f'end_{rep_num}: ' + str(time_))
    runtimelog.close()

    print(f'{rep_num} of {reps} concluded at {time_} \n')

    # want to save data_var, all_predicted_rakes, and pts_and_avg_rake to a file
    grp_pts_and_avg_rake.create_dataset(str(rep_num), data=pts_and_avg_rake)
    grp_all_pred_rake.create_dataset(str(rep_num), data=all_predicted_rakes)
    grp_data_var.create_dataset(str(rep_num), data=data_var)

    print('data_var outputted to file: ', str(file_name) + '.hdf5 in group ["data_var"] for data set ["', str(rep_num),
          '"]')
    print('all_predicted_rakes outputted to file: ',
          str(file_name) + '.hdf5 in group ["all_predicted_rakes"] for data set ["', str(rep_num), '"]')
    print('pts_and_avg_rake outputted to file: ',
          str(file_name) + '.hdf5 in group ["pts_and_avg_rake"] for data set ["', str(rep_num), '"]')

    f.close()

    # after storing data, reset variables
    pts_and_avg_rake, all_predicted_rakes, data_var = 0., 0., 0.

    # loop for total number of repetitions with data storing checkpoints, begins at last visited lattice location
    while rep_num < reps:
        rep_num += 1
        restart_loc = start_loc(slip_model[1], lattice, dictionaries, given=last_pt, mle=False)
        last_pt, pts_and_avg_rake, all_predicted_rakes, data_var, lr_ = mcmc_v01(rep_num, indiv_steps, lattice,
                                                                                 slip_model, uncertainties,
                                                                                 dictionaries, starting_loc=restart_loc,
                                                                                 lattice_nodes=[180, 10, 10],
                                                                                 save_to_file=file_name)

        LR_file = open(str(LRfile_name) + '.txt', 'a')
        LR_file.write(str(lr_) + '\n')
        LR_file.close()

        time_ = datetime.datetime.today().strftime("%H:%M:%S")
        runtimelog = open('runtimelog' + str(file_name) + '.txt', 'a')
        runtimelog.write(f'end_{rep_num}: ' + str(time_))
        runtimelog.close()
        print(f'{rep_num} of {reps} concluded at {time_}\n')

        # want to save data_var, all_predicted_rakes, and pts_and_avg_rake to a file
        f = h5py.File(str(file_name) + '_' + str(rep_num) + '.hdf5', 'w')
        grp_pts_and_avg_rake = f.create_group("pts_and_avg_rake")
        grp_all_pred_rake = f.create_group("all_predicted_rakes")
        grp_data_var = f.create_group("data_var")
        grp_pts_and_avg_rake.create_dataset(str(rep_num), data=pts_and_avg_rake)
        grp_all_pred_rake.create_dataset(str(rep_num), data=all_predicted_rakes)
        grp_data_var.create_dataset(str(rep_num), data=data_var)
        f.close()

        print('data_var outputted to file: ', str(file_name) + '.hdf5 in group ["data_var"] for data set ["',
              str(rep_num), '"]')
        print('all_predicted_rakes outputted to file: ',
              str(file_name) + '.hdf5 in group ["all_predicted_rakes"] for data set ["', str(rep_num), '"]')
        print('pts_and_avg_rake outputted to file: ',
              str(file_name) + '.hdf5 in group ["pts_and_avg_rake"] for data set ["', str(rep_num), '"]')

        pts_and_avg_rake, all_predicted_rakes, data_var, lr_ = 0., 0., 0., 0.


    return


"""

###########
#example run:
##########

initial_starting_locs = start_loc(test_faults,lattices,dictionaries_lattice_index,given='empty',mle=True,lattice_nodes=[180,50,50])

mcmc_implementation(10**2,lattices,initial_starting_locs,test_slip_model,test_uncertainties,dictionaries_lattice_index)

"""
