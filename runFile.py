# Example of an input file for computing stress posteriors on a given lattice for specified fault
# geometries and uncertainties

# Import libraries and modules:
import os
import h5py
import numpy as np

# Import stress inversion functions:
from stress_functions import *
from support_stress_functions import *

#MODIFY-BELOW----------------------------------------------
# Set length of RW (i.e., number of models retained), try O(1e6-7) for current lattice density. to test/debug, try O(1e2)
numSteps = 5 * 10 ** 2

# Names of files to use:
faultsFileName = 'testFaultGeometries.npy'
smFileName = 'testSM.npy'
rakeUncFileName = 'testRakeStdDev.npy'
faultUncFileName = 'testGeometryStdDev.npy'
jacobianFileName = 'testPrecomputedFx.h5'
#END--------------------------------------------------------

# Read in uncertainty files:
# fx_ : h5 file with keys for each model on lattice and values for precomputed Jacobians of prediction of rake equation
#       with respect to fault strike and fault dip; improves efficiency of error propagation calculations
# std_rake_ : numpy file with standard deviation on observed rake determined from coseismic slip model
# std_geometry_ : numpy file standard deviation on observed fault strike and fault dip determined from
#       coseismic slip model

fx_ = h5py.File('./uncertaintyFiles/'+jacobianFileName,'r')
std_rake_ = np.load('./uncertaintyFiles/'+rakeUncFileName,allow_pickle=True)
std_geometry_ = np.load('./uncertaintyFiles/'+faultUncFileName,allow_pickle=True)
uncertainties_ = [fx_, std_rake_, std_geometry_]

# Read in lattice and dictionaries:
# lattice_collection : numpy file for lattice dictionary
# ind_d : lattice index dictionary
# lattices : h5 file for lattice with 180 nodes along equator of (phi, theta) sphere sample space lattice,
#       180 nodes in wrapping rho lattice and 10 nodes each in delta and final-sigma lattices

lattice_collection = np.load('./latticeFiles/lattice_collection_dict.npy',allow_pickle=True).item()
ind_d = np.load('./latticeFiles/ind_d_dict.npy',allow_pickle=True).item()
lattices = h5py.File('./latticeFiles/lattices_sphere_1801010.h5','r')
dictionaries_lattice_index = [lattice_collection, ind_d]

# Read in slip model and fault files:
# faults_ : numpy file with fault geometries for the coseismic slip model
# sm_ : numpy file with _?_ for the coseismic slip model
faults_ = np.load('./faultGeometryFiles/'+faultsFileName,allow_pickle=True)
sm_ = np.load('./faultGeometryFiles/'+smFileName,allow_pickle=True)
slip_model_ = [sm_, faults_]

# Move to directory where posteriors and output will be saved
os.chdir("./stressPosteriors")

# Compute initial starting location on lattice; starts random walk near P-axis to reduce burn-in
initial_starting_locs = start_loc(faults_, lattices, dictionaries_lattice_index, given='empty', mle=True,
                                  lattice_nodes=[180, 10, 10])

# Implement MCMC; output saves stress posterior as h5 files to stressPosteriors directory that can be
#       post-processed for plotting principal stress components as lower hemisphere Lambert projections
#       and marginals for relative magnitudes (delta, finalSigma) as histograms

RW = mcmc_implementation(numSteps, lattices, initial_starting_locs, slip_model_, uncertainties_,
                                  dictionaries_lattice_index)
