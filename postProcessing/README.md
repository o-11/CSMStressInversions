### 1. Export check-point files

After computing the Bayesian MCMC posterior from the slip model by running runFile.py in the main directory, the outputted check-point files saved to the stressPosteriors directory must be exported to a single file. You will need to modify the saved file name prefix and the length of the random walk specified by runFile.py.

In terminal, run `python exporth5.py`

### 2. Compute symmetries, principal components, and lower hemisphere Lambert projection of principal components of stress
Modify the name of the exported file name from (1) above in the findSymmetries.py file.
Then, you will need to run the findSymmetries.py, findPointsfromIndices.py, and findEigenfromPoints.py scripts. 

In terminal, run `python findSymmetries.py`
`python findPointsfromIndices.py`
and
`python findEigenfromPoints.py` 

###3. Create and save plots
To create lower hemisphere Lambert projection plots of the principal components of stress (MCS, ICS, and LCS), in terminal run:

`python saveLowerHemispherePlots.py` 

To create rose plots of the predicted vs. observed rake, run the following in terminal:

`python saveRosePlots.py`

All plots will be saved to the Plots subdirectory.

### (optional) Computing composite posteriors from multiple slip models of the same event
Following step (1) for all of the individual posteriors of interest, if you wish to compute the non-mutually exclusive union composite posteriors from the posterior of multiple coseismic slip models, modify the file names in computeCompositePosterior.py script within this directory and then run in python through the terminal as:

`python computeCompositePosterior.py`

You will then need to perform steps (2) and (3) above.