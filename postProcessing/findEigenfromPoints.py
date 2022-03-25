import numpy as np
import h5py

#Modify-------------------------------
name='test_'
#End----------------------------------

point_arr=np.load(name+'idxconverted_points.npy',allow_pickle=True)
freq = np.load(name+'counts_with_symmetries.npy',allow_pickle=True)

#-----
latt_file=h5py.File('../latticeFiles/lattices_sphere_1801010.h5','r')
sph=latt_file['sph_nodes'][:]

#-----Functions------
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


def find_nearest_point(point_arr, point): #points on surface of a sphere
    """
    ###
    # find_nearest_point: find index of point and point within array that is nearest to given point, uses a
ngle_between function
    ###
    """

    idx = (angle_between(point_arr, point)).argmin()
    return idx, point_arr[idx]


def stress_rotation_to_global(ϕ, θ, ρ):
    """
    ###
    # stress_rotation_to_global: rotate stress tensor to global coordinates
    ###
    # rotates stress into global coordinates
    """

    rot = np.matrix([[np.cos(θ) * np.cos(ϕ), -np.cos(ϕ) * np.sin(θ) * np.sin(ρ) - np.cos(ρ) * np.sin(ϕ),
                      -np.cos(ρ) * np.cos(ϕ) * np.sin(θ) + np.sin(ρ) * np.sin(ϕ)],
                     [np.cos(θ) * np.sin(ϕ), np.cos(ρ) * np.cos(ϕ) - np.sin(θ) * np.sin(ρ) * np.sin(ϕ),
                      -np.cos(ϕ) * np.sin(ρ) - np.cos(ρ) * np.sin(θ) * np.sin(ϕ)],
                     [np.sin(θ), np.cos(θ) * np.sin(ρ), np.cos(θ) * np.cos(ρ)]])
    return rot


#-------------------------------------------------------
def spherical_to_cartesian(point):

    """
    ###
    # spherical_to_cartesian: assuming S2 sphere with radius = 1, convert spherical coords to cartesian coords
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

#-------------------------------------------------------
def eig_stress(points):

    """
    calculate mcs,ics and lcs from rw through mcs as eigen-system from retained model 'points' from rw:
    """

    mcs = np.zeros((len(points), 2));
    ics = np.zeros((len(points), 2));
    lcs = np.zeros((len(points), 2))
    σ1_mag = -1
    for j in range(len(points)):
        ϕ_, θ_, ρ_, Δ_, R_ = points[j, 0:5]
        R_g = stress_rotation_to_global(ϕ_, θ_, ρ_)  # form rotation matrix for global coordinates;
    #σ_reference = np.array([[σ1_mag, 0, 0], [0, Δ_ * (σ1_mag - R_ * σ1_mag) + (R_ * σ1_mag), 0],
                                 #[0, 0, R_ * σ1_mag]])  # reference stress tensor;
    #eig_vals = np.diag(σ_reference)
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

        mcs[j,0] = trend[0]; mcs[j,1] = plunge[0]
        ics[j,0] = trend[1]; ics[j,1] = plunge[1]
        lcs[j,0] = trend[2]; lcs[j,1] = plunge[2]

    return np.array([mcs, ics, lcs])

#----------------------------------------------
def mirror_points_2(cartesianpoint):
    """
    #mirrors points to the lower hemisphere for plotting and for equatorial symmetry
    """
    return np.array([cartesianpoint[i] if cartesianpoint[i,2]<0. else -1*cartesianpoint[i] for i in range(len(cartesianpoint))])


def cartesian_mirrors(eigvecs):
    """
    #calculate cartesian, mirror points (projects to lower hemisphere), equatorial mirrorring (symmetry for z=0), rotate:
    """

    m_pts, i_pts, l_pts = eigvecs

    del eigvecs

    m_cartesian = spherical_to_cartesian(m_pts);
    m_round = np.round(m_cartesian,decimals=6);
    m_mirror = mirror_points_2(m_round);
    m_rot = np.zeros((len(m_mirror),3));
    del m_pts; del m_cartesian; del m_round

    i_cartesian = spherical_to_cartesian(i_pts);
    i_round = np.round(i_cartesian,decimals=6);
    i_mirror = mirror_points_2(i_round);
    i_rot = np.zeros((len(i_mirror),3));
    del i_pts; del i_cartesian; del i_round

    l_cartesian = spherical_to_cartesian(l_pts);
    l_round = np.round(l_cartesian,decimals=6);
    l_mirror = mirror_points_2(l_round);
    l_rot = np.zeros((len(l_mirror),3));
    del l_pts; del l_cartesian; del l_round
    #rotation required to go from seismic convention cartesian coordinates (north(+x),east(-y),down(+z)) to map coordinates (north(+y),east(+x),up(+z)) for lambert projection plot visualizations of MCS,ICS and LCS; single for loop for efficiency
    max_len=max(len(m_rot),len(i_rot),len(l_rot))
    rot_mat=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    for j in range(max_len):
        try:
            m_rot[j] = rot_mat @ m_mirror[j]
        except:
            continue
        try:
            i_rot[j] = rot_mat @ i_mirror[j]
        except:
            continue
        try:
            l_rot[j] = rot_mat @ l_mirror[j]
        except:
            continue

    return [m_rot,i_rot,l_rot]

#-------------------------------------------------------
def lambert_transformation(mirrorpoint):
    # projects points into equal area lower-hemisphere for plotting
    x = mirrorpoint[:, 0];
    y = mirrorpoint[:, 1];
    z = mirrorpoint[:, 2];
    easting = [];
    northing = [];
    for i in range(len(mirrorpoint)):
        if np.allclose(mirrorpoint[i], [0, 0, 1]):
            X = 0;
            Y = 0;
        else:
            X = (2 / (1 + z[i])) ** (0.5) * x[i];
            Y = (2 / (1 + z[i])) ** (0.5) * y[i];
        easting.append(X);
        northing.append(Y);
    projection = np.array([easting, northing]).T;

    return projection


# -------------------------------------------------------
def lambert_points(rotatedpoints):
    """
    #lambert projection points
    """

    m_projection_unique = np.round(lambert_transformation(rotatedpoints[0]), decimals=6);
    i_projection_unique = np.round(lambert_transformation(rotatedpoints[1]), decimals=6);
    l_projection_unique = np.round(lambert_transformation(rotatedpoints[2]), decimals=6);

    return [m_projection_unique, i_projection_unique, l_projection_unique]

# -----Eig-Stress-----
eigvecs_ = eig_stress(point_arr)
m,i,l=eigvecs_
print('eigenvectors calculated')

mcs_unique_pts,mcs_unique_ind,mcs_unique_counts=np.unique(m,return_index=True,return_counts=True,axis=0)

freqs_m1=np.zeros((len(mcs_unique_ind),1))
for k in range(len(mcs_unique_ind)):
    xs=np.where(m[:,0]==m[mcs_unique_ind][k,0])[0]
    ys=np.where(m[xs,1].T==m[mcs_unique_ind][k,1])[0]
    points=m[xs[ys]]
    freqs_m1[k]=np.sum(freq[xs[ys]])

new_m_latt_pts=np.zeros((len(mcs_unique_pts),2))
for mcs in range(len(mcs_unique_pts)):
    new_m_latt_pts[mcs]=find_nearest_point(sph,mcs_unique_pts[mcs])[1]

print('mcs freqs 1 calculated')

new_i_latt_pts=np.zeros((len(i),2))
for ics in range(len(i)):
    new_i_latt_pts[ics]=find_nearest_point(sph,i[ics])[1]

ics_unique_pts,ics_unique_ind,ics_unique_counts=np.unique(new_i_latt_pts,return_index=True,return_counts=True,axis=0)
print('ics unique 1 calculated')

freqs_i=np.zeros((len(ics_unique_ind),1))
for k in range(len(ics_unique_ind)):
    xs=np.where(new_i_latt_pts[:,0]==new_i_latt_pts[ics_unique_ind][k,0])[0]
    ys=np.where(new_i_latt_pts[xs,1].T==new_i_latt_pts[ics_unique_ind][k,1])[0]
    freqs_i[k]=np.sum(freq[xs[ys]])
print('ics freqs 1 calculated')

new_l_latt_pts=np.zeros((len(l),2))
for lcs in range(len(l)):
    new_l_latt_pts[lcs]=find_nearest_point(sph,l[lcs])[1]

lcs_unique_pts,lcs_unique_ind,lcs_unique_counts=np.unique(new_l_latt_pts,return_index=True,return_counts=True,axis=0)
print('lcs unique 1 calculated')

freqs_l=np.zeros((len(lcs_unique_ind),1))
for k in range(len(lcs_unique_ind)):
    xs=np.where(new_l_latt_pts[:,0]==new_l_latt_pts[lcs_unique_ind][k,0])[0]
    ys=np.where(new_l_latt_pts[xs,1].T==new_l_latt_pts[lcs_unique_ind][k,1])[0]
    freqs_l[k]=np.sum(freq[xs[ys]])
print('lcs freqs 1 calculated')

mapped_eigvecs_ = [new_m_latt_pts,ics_unique_pts,lcs_unique_pts]

# -----Rotation------
rot_cart_ = cartesian_mirrors(mapped_eigvecs_)

# -----Lambert_Points---
lambpts_ = lambert_points(rot_cart_)
#np.save(name + 'lambert_points.npy', lambpts_, allow_pickle=True)
#print('lambert points saved')

m_lp,i_lp,l_lp=lambpts_
np.save(str(name)+'_lps.npy',lambpts_,allow_pickle=True)
print('lambert points calculated and saved')

mcs_unique_pts,mcs_unique_ind,mcs_unique_counts=np.unique(m_lp,return_index=True,return_counts=True,axis=0)

freqs_m_=np.zeros((len(mcs_unique_ind),1))
for k in range(len(mcs_unique_ind)):
    xs=np.where(m_lp[:,0]==m_lp[mcs_unique_ind][k,0])[0]
    ys=np.where(m_lp[xs,1].T==m_lp[mcs_unique_ind][k,1])[0]
    points=m_lp[xs[ys]]
    freqs_m_[k]=np.sum(freqs_m1[xs[ys]])

mcs_cmap_pts_=freqs_m_.T[0]
np.save(str(name)+'_mcs_cmap.npy',mcs_cmap_pts_,allow_pickle=True)
np.save(str(name)+'_mcs_pts.npy',mcs_unique_pts,allow_pickle=True)
print('mcs cmap and pts saved')

ics_unique_pts,ics_unique_ind,ics_unique_counts=np.unique(i_lp,return_index=True,return_counts=True,axis=0)

freqs_all_i=np.zeros((len(ics_unique_ind),1))
for k in range(len(ics_unique_ind)):
    xs=np.where(i_lp[:,0]==i_lp[ics_unique_ind][k,0])[0]
    ys=np.where(i_lp[xs,1].T==i_lp[ics_unique_ind][k,1])[0]
    points=i_lp[xs[ys]]
    freqs_all_i[k]=np.sum(freqs_i[xs[ys]])

ics_cmap_pts_=freqs_all_i.T[0]
np.save(str(name)+'_ics_cmap.npy',ics_cmap_pts_,allow_pickle=True)
np.save(str(name)+'_ics_pts.npy',ics_unique_pts,allow_pickle=True)
print('ics pts and cmap saved')

lcs_unique_pts,lcs_unique_ind,lcs_unique_counts=np.unique(l_lp,return_index=True,return_counts=True,axis=0)

freqs_all_l=np.zeros((len(lcs_unique_ind),1))
for k in range(len(lcs_unique_ind)):
    xs=np.where(l_lp[:,0]==l_lp[lcs_unique_ind][k,0])[0]
    ys=np.where(l_lp[xs,1].T==l_lp[lcs_unique_ind][k,1])[0]
    points=l_lp[xs[ys]]
    freqs_all_l[k]=np.sum(freqs_l[xs[ys]])

lcs_cmap_pts_=freqs_all_l.T[0]

np.save(str(name)+'_lcs_cmap.npy',lcs_cmap_pts_,allow_pickle=True)
np.save(str(name)+'_lcs_pts.npy',lcs_unique_pts,allow_pickle=True)
print('lcs pts and cmap saved')
