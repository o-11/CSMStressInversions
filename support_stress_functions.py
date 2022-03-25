import numpy as np

def spherical_to_cartesian(point):

    """
    ###
    # spherical_to_cartesian: assuming S2 sphere with radius = 1, convert spherical coordinates to cartesian
    ###
    """
    
    try:
        trend = point[:, 0];
        plunge = point[:, 1]
    except:
        trend, plunge = point
    x = np.cos(trend) * np.cos(plunge)
    y = np.sin(trend) * np.cos(plunge)
    z = np.sin(plunge)
    cartesianpoint = np.array([x, y, z]).T

    return cartesianpoint


def cartesian_to_spherical(point):

    """
    ###
    # cartesian_to_spherical: assuming S2 sphere with radius = 1, convert cartesian coordinates to spherical
    ###
    """
    try:
        x = point[:, 0];
        y = point[:, 1];
        z = point[:, 2]
    except:
        x = point[0];
        y = point[1];
        z = point[2]
    trend = np.arctan2(y, x)
    plunge = np.arctan2(z, (x ** 2 + y ** 2) ** 0.5)
    sphericalpoint = np.array([trend, plunge]).T

    return sphericalpoint


def stress_rotation_to_fault(Φ, Θ, Λ=0):

    """
    ###
    # stress_rotation_to_fault: rotate stress tensor to fault coordinates
    ###
    """
    
    Θ_C = -np.pi / 2 + Θ;
    rot = np.matrix([[-np.cos(Φ), np.sin(Φ), 0], [np.cos(Θ_C) * np.sin(Φ), np.cos(Θ_C) * np.cos(Φ), -np.sin(Θ_C)],
                     [np.sin(Θ_C) * np.sin(Φ), np.sin(Θ_C) * np.cos(Φ), np.cos(Θ_C)]]);
    return rot


def stress_rotation_to_global(ϕ, θ, ρ):
    """
    ###
    # stress_rotation_to_global: rotate stress tensor to global coordinates
    ###
    """
    
    rot = np.matrix([[np.cos(θ) * np.cos(ϕ), -np.cos(ϕ) * np.sin(θ) * np.sin(ρ) - np.cos(ρ) * np.sin(ϕ),
                      -np.cos(ρ) * np.cos(ϕ) * np.sin(θ) + np.sin(ρ) * np.sin(ϕ)],
                     [np.cos(θ) * np.sin(ϕ), np.cos(ρ) * np.cos(ϕ) - np.sin(θ) * np.sin(ρ) * np.sin(ϕ),
                      -np.cos(ϕ) * np.sin(ρ) - np.cos(ρ) * np.sin(θ) * np.sin(ϕ)],
                     [np.sin(θ), np.cos(θ) * np.sin(ρ), np.cos(θ) * np.cos(ρ)]])
    return rot


def fault_plane_to_moment_tensor(Φ, Θ, Λ, moment=1):
    """
    ###
    # fault_plane_to_moment_tensor: convert fault plane strike,dip, rake and moment to a cartesian moment tensor in Aki&Richards 1984 convention
    ###
    
    # Cartesian moment tensor, based on Aki&Richards 1984 conventions;
    ##Φ: Fault Strike, Θ: Fault Dip, Λ: Fault Rake
    """
    
    M = np.zeros((3, 3));
    M[0, 0] = -moment * (np.sin(Θ) * np.cos(Λ) * np.sin(2 * Φ) + np.sin(2 * Θ) * np.sin(Λ) * (np.sin(Φ) ** 2))
    M[0, 1] = moment * (np.sin(Θ) * np.cos(Λ) * np.cos(2 * Φ) + (1 / 2) * np.sin(2 * Θ) * np.sin(Λ) * np.sin(2 * Φ))
    M[0, 2] = -moment * (np.cos(Θ) * np.cos(Λ) * np.cos(Φ) + np.cos(2 * Θ) * np.sin(Λ) * np.sin(Φ))
    M[1, 1] = moment * (np.sin(Θ) * np.cos(Λ) * np.sin(2 * Φ) - np.sin(2 * Θ) * np.sin(Λ) * (np.cos(Φ) ** 2))
    M[1, 2] = -moment * (np.cos(Θ) * np.cos(Λ) * np.sin(Φ) - np.cos(2 * Θ) * np.sin(Λ) * np.cos(Φ))
    M[2, 2] = moment * np.sin(2 * Θ) * np.sin(Λ)
    M[1, 0] = M[0, 1]
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]
    return M


def moment_tensor_to_pnt(mt):
    """
    ###
    # moment_tensor_to_pnt: convert moment tensor to p,n,t axes
    ###
    """

    λ, v = np.linalg.eig(mt)
    stack = np.vstack([λ, v])
    λ_sort_p = np.round(stack[:, stack[0, :].argsort()][0, :], decimals=6)
    λ_sort = λ_sort_p
    v_sort_p = np.round(stack[:, stack[0, :].argsort()][1:4, :], decimals=6)
    v_sort = np.vstack((v_sort_p[0], v_sort_p[1], v_sort_p[2]))

    azi = np.zeros((3));
    dip = np.zeros((3))
    for i in range(3):
        if v_sort[0, i] == v_sort[1, i] == 0:
            azi[i] = 0.0
        else:
            azi[i] = np.arctan2(v_sort[1, i], v_sort[0, i])
        dip[i] = np.arctan2(v_sort[2, i], np.sqrt(v_sort[0, i] ** 2 + v_sort[1, i] ** 2))
        if dip[i] > np.pi / 2:
            dip[i] = dip[i] - np.pi
        if dip[i] < 0:
            azi[i] = azi[i] + np.pi
            dip[i] = -dip[i]
        if azi[i] > np.pi:
            azi[i] = azi[i] - 2 * np.pi
        if azi[i] < -np.pi:
            azi[i] = azi[i] + 2 * np.pi

    p = np.array([azi[0], dip[0]])
    n = np.array([azi[1], dip[1]])
    t = np.array([azi[2], dip[2]])

    return p, n, t, v_sort


def angle_between(a, b):
    """
    ###
    # angle_between: takes two spherical coordinates and determines the angle between them
    ###
    """
    
    a = spherical_to_cartesian(a);
    b = spherical_to_cartesian(b);
    θ = np.arccos((np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)));
    return θ


def find_nearest_point(point_arr, point):
    """
    ###
    # find_nearest_point: find index of point and point within array that is nearest to given point, uses angle_between function
    ###
    """
    
    idx = (angle_between(point_arr, point)).argmin()
    return idx, point_arr[idx]


def wtd_ang_avg(a, p):
    """
    ###
    # wtd_ang_avg: Calculate the weighted angular average from an array of angles, a, with weights, p:
    ###
    """
    p_sin = np.zeros((len(a)))
    p_cos = np.zeros((len(a)))
    for i in range(len(a)):
        p_sin[i] = p[i] * np.sin(a[i])
        p_cos[i] = p[i] * np.cos(a[i])
    avg = np.arctan2(1 / np.sum(p) * np.sum(p_sin), 1 / np.sum(p) * np.sum(p_cos))
    return avg


def wtd_ang_var(a, p):
    """
    # expression originally developed by me
    ###
    # wtd_ang_var: Calculate the weighted variance from an array of angles, a, with weights, p:
    ###
    """
    n = len(a)
    α = wtd_ang_avg(a, p)
    return np.arctan2((1 / (np.sum(p)) * np.sum(p * np.sin(a - α) ** 2)),
                      (1 / (np.sum(p)) * np.sum(p * np.cos(a - α) ** 2)))


def fault_wtd_avg_params(slipmodel):
    """
    ###
    # fault_wtd_avg_params: Generate default format for fault parameters from a given slip model for RW
    ###
    """
    
    num_faults = int(np.max(slipmodel[:, 0]) + 1)
    faults = np.zeros((num_faults, 6))
    sub_fault_pot = np.zeros((len(slipmodel), 2))
    seg_pot = np.zeros((num_faults, 1))
    var_Φ = np.zeros((num_faults, 1))
    var_Θ = np.zeros((num_faults, 1))

    sub_fault_pot[:, 0] = slipmodel[:, 0]
    for p in range(len(slipmodel)):
        sub_fault_pot[p, 1] = (slipmodel[p, 4]) * (slipmodel[p, 5]) * (slipmodel[p, 6])

    for f in range(num_faults):
        faults[f, 0] = wtd_ang_avg(slipmodel[np.where(slipmodel[:, 0] == f), 1][0],
                                   sub_fault_pot[np.where(sub_fault_pot[:, 0] == f), 1][
                                       0])  # avg strike on fault seg, wtd by potency
        faults[f, 1] = wtd_ang_avg(slipmodel[np.where(slipmodel[:, 0] == f), 2][0],
                                   sub_fault_pot[np.where(sub_fault_pot[:, 0] == f), 1][
                                       0])  # avg dip on fault seg, wtd by potency
        faults[f, 2] = wtd_ang_avg(slipmodel[np.where(slipmodel[:, 0] == f), 3][0],
                                   sub_fault_pot[np.where(sub_fault_pot[:, 0] == f), 1][
                                       0])  # avg rake on fault seg, wtd by potency

        faults[f, 3] = np.sum(slipmodel[np.where(slipmodel[:, 0] == f), 4][0])  # total sum of slip on segment; 
        faults[f, 4] = np.sum(slipmodel[np.where(slipmodel[:, 0] == f), 5][0])  # total sum of length on segment; this is not the same as the length of the fault! Will either be equal to or greater than
        faults[f, 5] = np.sum(slipmodel[np.where(slipmodel[:, 0] == f), 6][0])  # total sum of width on segment; this is not the same as the width of the fault! Will either be equal to or greater than

        seg_pot[f] = np.sum(sub_fault_pot[np.where(sub_fault_pot[:, 0] == f), 1])

    return faults, seg_pot, num_faults


def lattice_to_index(ϕ,θ,ρ,Δ,lc,index_dict):

    ϕθ = str(ϕ) + '_' + str(θ)
    id_ = str(lc["ϕθ"][ϕθ]+'_'+lc["ρ"][ρ]+'_'+lc["Δ"][Δ])

    return int(index_dict[id_])-1


def get_fx_for_sm(all_fx_arrays_for_sm, num_faults, str_dict, ind_dict, model):

    """
    function to find fx value for model step visited for all fault planes within slip model;
    all_fx_arrays_for_sm = np.hstack((fx_fault_1,fx_fault_2....fx_fault_num_faults)), form outside of simulation;
    str_dict = ; ind_dict = ; dependent upon num nodes
    """

    index_ = lattice_to_index(model[0],model[1],model[2],model[3],str_dict,ind_dict)
    fx_for_model = [all_fx_arrays_for_sm['fault_'+str(f)][index_] for f in range(1,num_faults+1)]

    return fx_for_model


def σ_P_(fx_model,geom_unc):
    """
    calculates covariance diagonal (variance of predicted rake) as the error propagation of strike and     dip uncertainty (σ_Φ,σ_Θ) from the precomputed fx values that correspond to the current model
    """
    return np.array([np.sqrt(fx_model[f][0] ** 2 * geom_unc[f,0] ** 2 + fx_model[f][1] ** 2 * geom_unc[f,1] ** 2) for f in range(0,len(fx_model))]).T


def predict_rake(faults, model, num_faults):

    """
    predict rake from a multi-fault slip model for a particular step in RW:
    """
    
    ϕ_, θ_, ρ_, Δ_, R_ = model
    σ1_mag = -1
    σ_reference = np.matrix([[σ1_mag, 0, 0], [0, Δ_ * (σ1_mag - R_ * σ1_mag) + (R_ * σ1_mag), 0],
                             [0, 0, R_ * σ1_mag]])  # reference stress tensor;
    R_g = stress_rotation_to_global(ϕ_, θ_, ρ_)  # form rotation matrix for global coordinates;
    σ_g = R_g @ σ_reference @ R_g.T  # rotate reference stress state into global (N,E,Z) coordinates

    pred_λ = np.zeros((num_faults,1));
    CS = np.zeros((num_faults,1))

    for f in range(num_faults):
        R_f = stress_rotation_to_fault(faults[f, 0], faults[f, 1], 0)  # form rotation matrix for in-fault coordinates;
        σ_f = R_f @ σ_g @ R_f.T  # rotate global stress tensor into fault

        strike_shear = σ_f[0, 1]  # shear stress along strike
        dip_shear = σ_f[2, 1]  # shear stress along dip

        pred_λ[f] = np.arctan2(dip_shear, strike_shear)  # predicted rake is co-linear to maximum shear stress

        σ_N = σ_f[1, 1]  # stress normal to fault plane
        τ = -np.sqrt(strike_shear ** 2 + dip_shear ** 2)  # shear stress on fault plane
        CS[f] = τ / σ_N  #

    return pred_λ, CS


def Gauss_Ll_prop(num_faults_, seg_pot_, obs_λ_, pred_λ_, std_λ__, σ_P_model_):

    """
    calculate exponential term of Gaussian likelihood for test model prediction w/ rake and error propagated uncertainties
    """
    
    σ_total = np.zeros((num_faults_,1))
    for f in range(num_faults_):
        σ_total[f] = min(std_λ__[f]**2 + σ_P_model_[f]**2, 2*np.pi)
        
    σ_total_inv = np.diag(1 / σ_total.T[0])
    LL = (- 1 / 2) * (pred_λ_ - obs_λ_).T @ (σ_total_inv) @ (pred_λ_ - obs_λ_)
    
    return LL[0], σ_total.T[0]


def eig_stress(points):

    """
    calculate mcs,ics and lcs from rw through mcs as eigen-system from retained model 'points' from rw
    finding eigenvectors as columns of stress tensor rotated into global coords is more efficient than solving eigen-system
    """

    mcs = np.zeros((len(points), 2));
    ics = np.zeros((len(points), 2));
    lcs = np.zeros((len(points), 2))
    for j in range(len(points)):
        ϕ_, θ_, ρ_, Δ_, R_ = points[j, 0:5]
        R_g = stress_rotation_to_global(ϕ_, θ_, ρ_)  # form rotation matrix for global coordinates;

        # not necessary for calculating vectors so commented out:
        # σ1_mag = -1
        # σ_reference = np.matrix([[σ1_mag, 0, 0], [0, Δ_ * (σ1_mag - R_ * σ1_mag) + (R_ * σ1_mag), 0], [0, 0, R_ * σ1_mag]])  # reference stress tensor;
        # eig_vals = np.diag(σ_reference)

        eig_vecs = np.round(R_g, decimals=6)
        trend = np.zeros((3));
        plunge = np.zeros((3))
        for k in range(3):
            trend[k] = np.arctan2(eig_vecs[1, k], eig_vecs[0, k])
            plunge[k] = np.arctan2(eig_vecs[2, k], np.sqrt(eig_vecs[0, k] ** 2 + eig_vecs[1, k] ** 2))
            if plunge[k] < 0:
                trend[k] += np.pi
                plunge[k] *= -1
            if plunge[k] > np.pi / 2:
                plunge[k] -= np.pi
            if plunge[k] < 0:
                trend[k] += np.pi
                plunge[k] *= -1
            if trend[k] > np.pi:
                trend[k] -= 2 * np.pi
            if trend[k] < -np.pi:
                trend[k] += 2 * np.pi

            trend[k] = np.round(trend[k], decimals=6)
            plunge[k] = np.round(plunge[k], decimals=6)

        mcs[j] = np.array([trend[0], plunge[0]])
        ics[j] = np.array([trend[1], plunge[1]])
        lcs[j] = np.array([trend[2], plunge[2]])

    return np.array([mcs, ics, lcs])
