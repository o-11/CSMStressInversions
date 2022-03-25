For new fault geometries, precompute the Jacobians for the prediction of rake w.r.t. fault strike and fault dip for the complete lattice. 
These values are used in computing the error propagated into the predicted rake for uncertain fault strike and fault dip for each stress model sampled (i.e., lattice node visited).

In runPrecomputeFx.py, modify the output file name and fault geometry for precompute_fx. The fault geometry is the fault strike in radians followed by the fault dip in radians.

To run in terminal, run `python runPrecomputeFx.py`