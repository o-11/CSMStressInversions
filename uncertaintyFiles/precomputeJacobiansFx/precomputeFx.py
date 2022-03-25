import datetime
import numpy as np
import time

all_nodes_180 = np.load('../../latticeFiles/all_nodes_latt180.npy',allow_pickle=True)

# calculate jacobian of Φ and Θ ('fx') for given fault strike, fault dip, and lattice location - more heavily optimized
def fxOpt2(CphiF,SphiF,CthetaF,SthetaF,ϕ,θ,ρ,Δ):

    Cphi = np.cos(ϕ)
    Sphi = np.sin(ϕ)

    Ctheta = np.cos(θ)
    Stheta = np.sin(θ)

    Crho = np.cos(ρ)
    Srho = np.sin(ρ)

    delta = (-0.5*Δ - 0.5)

    term1 = (-Stheta*Srho*Cphi-Sphi*Crho)
    term2 = (-Stheta*Srho*Sphi+Crho*Cphi)
    term3 = (-Stheta*Crho*Cphi+Srho*Sphi)
    term4 = (-Stheta*Sphi*Crho-Srho*Cphi)
    #term5 = (-Stheta*Srho*Sphi+Crho*Cphi)

    c1 = (delta*term2*Srho*Ctheta-0.5*term4*Ctheta*Crho-Stheta*Sphi*Ctheta)
    c2 = (-delta*term1*Srho*Ctheta+0.5*term3*Ctheta*Crho+Stheta*Ctheta*Cphi)
    c3 = (delta*term1*Srho*Ctheta-0.5*term3*Ctheta*Crho-Stheta*Ctheta*Cphi)

    b1= ((delta*Srho**2*Ctheta**2-Stheta**2-0.5*Ctheta**2*Crho**2)*SthetaF-c1*CthetaF*CphiF-c3*SphiF*CthetaF)*CthetaF
    b2 = -(delta*term2**2-0.5*term4**2-Sphi**2*Ctheta**2)*CthetaF*CphiF-(delta*term2*term1-0.5*term4*term3-Sphi*Ctheta**2*Cphi)*SphiF*CthetaF

    bar1 = (delta*term1**2-0.5*term3**2-Ctheta**2*Cphi**2)
    bar2 = (delta*term2*term1-0.5*term4*term3-Sphi*Ctheta**2*Cphi)
    bar3 = (delta*term2**2-0.5*term4**2-Sphi**2*Ctheta**2)
    bar4 = (c1*SphiF-c3*CphiF)

    c = (bar4*CthetaF + (-bar1 + bar3)*CphiF*SphiF*SthetaF + bar2*(-CphiF**2 + SphiF**2)*SthetaF)**2 + (b1 + SthetaF*(b2*CphiF - CthetaF*SphiF*(bar2*CphiF + bar1*SphiF) + (c1*CphiF + c3*SphiF)*SthetaF))**2

    a_jΦ = (1/c)*((b1 + SthetaF*(b2*CphiF - CthetaF*SphiF*(bar2*CphiF + bar1*SphiF) + (c1*CphiF + c3*SphiF)*SthetaF))*(c1*CphiF*CthetaF - c2*CthetaF*SphiF + (-bar1 + bar3)*CphiF**2*SthetaF + SphiF*SthetaF*(3.*bar2*CphiF -Cphi*CphiF*Ctheta**2*Sphi - bar3*SphiF - Cphi**2*Ctheta**2*SphiF + delta*SphiF*term1**2 + CphiF*delta*term1*term2 - 0.5*SphiF*term3**2 - 0.5*CphiF*term3*term4)) + (-bar4*CthetaF + (bar1 - bar3)*CphiF*SphiF*SthetaF + bar2*(CphiF - SphiF)*(CphiF + SphiF)*SthetaF)*(-SphiF*SthetaF*(b2 + c1*SthetaF) - CphiF*SthetaF*(bar2*CphiF*CthetaF + bar1*CthetaF*SphiF - c3*SthetaF) + CthetaF*SphiF*SthetaF*(Cphi**2*CphiF*Ctheta**2 - Cphi*Ctheta**2*Sphi*SphiF - CphiF*delta*term1**2 + delta*SphiF*term1*term2 + 0.5*CphiF*term3**2 -0.5*SphiF*term3*term4) +CphiF*CthetaF*SthetaF*(Cphi*CphiF*Ctheta**2*Sphi -Ctheta**2*Sphi**2*SphiF - CphiF*delta*term1*term2 + delta*SphiF*term2**2 + 0.5*CphiF*term3*term4 - 0.5*SphiF*term4**2) + CthetaF**2*(c2*CphiF + Ctheta*SphiF*(-Sphi*Stheta + delta*Srho*term2 - 0.5*Crho*term4))))

    a_jΘ = (1/c)*(((-bar1 + bar3)*CphiF*CthetaF*SphiF + bar2*CthetaF*(-CphiF**2 + SphiF**2) - bar4*SthetaF)*(b1 + SthetaF*(b2*CphiF - CthetaF*SphiF*(bar2*CphiF + bar1*SphiF) + (c1*CphiF + c3*SphiF)*SthetaF)) + (bar4*CthetaF + (-bar1 + bar3)*CphiF*SphiF*SthetaF + bar2*(-CphiF**2 + SphiF**2)*SthetaF)*(-b2*CphiF*CthetaF + bar2*CphiF*CthetaF**2*SphiF + bar1*CthetaF**2*SphiF**2 - Ctheta**2*CthetaF**2*delta*Srho**2 + CthetaF**2*Stheta**2 - 3.*c1*CphiF*CthetaF*SthetaF + c2*CthetaF*SphiF*SthetaF - 3.*c3*CthetaF*SphiF*SthetaF + CphiF*Ctheta*CthetaF*Sphi*Stheta*SthetaF + CphiF**2*Ctheta**2*Sphi**2*SthetaF**2 + 2.*Cphi*CphiF*Ctheta**2*Sphi*SphiF*SthetaF**2 + Cphi**2*Ctheta**2*SphiF**2*SthetaF**2 + Ctheta**2*delta*Srho**2*SthetaF**2 - Stheta**2*SthetaF**2 + Crho**2*Ctheta**2*(0.5*CthetaF**2 - 0.5*SthetaF**2) - delta*SphiF**2*SthetaF**2*term1**2 - CphiF*Ctheta*CthetaF*delta*Srho*SthetaF*term2 - 2.*CphiF*delta*SphiF*SthetaF**2*term1*term2 - CphiF**2*delta*SthetaF**2*term2**2 + 0.5*SphiF**2*SthetaF**2*term3**2 + 0.5*CphiF*Crho*Ctheta*CthetaF*SthetaF*term4 + CphiF*SphiF*SthetaF**2*term3*term4 + 0.5*CphiF**2*SthetaF**2*term4**2))

    return np.array([a_jΦ, a_jΘ]).T



# precompute fx for all possible lattice locations for a given fault;
## fault_name: string for file save prefix to indicate fault
## fault: array with first element fault strike and second element fault dip
## arr_of_nodes: either all_nodes_180 or all_nodes_360
## num_nodes: typically len(arr_of_nodes)
### example 1. fx_for_360_nodes = precompute_fx('test_fault_1',np.array([0.0,np.pi/2]),all_nodes_360,len(all_nodes_360));
### example 2. fx_for_180_nodes = precompute_fx('test_fault_1',np.array([0.0,np.pi/2]),all_nodes_180,len(all_nodes_180));
def precompute_fx(fault_name,fault,arr_of_nodes,num_nodes):


    Φ,Θ=fault[0:2]
    fxO_=np.zeros((num_nodes,2))

    cPhiF = np.cos(Φ)
    sPhiF = np.sin(Φ)

    cThetaF = np.cos(Θ)
    sThetaF = np.sin(Θ)

    check = 1000000
    outernum = np.ceil(num_nodes/check).astype(int)

    k=-1
    outer_time = time.time()
    for outer in range(outernum):
        inner_time = time.time()

        if outer<outernum-1:
            thisnum = check
        else:
            thisnum = num_nodes-(outernum-1)*check

        for inner in range(thisnum):
            k = k+1
            ϕ, θ, ρ, Δ=arr_of_nodes[k]

            fxO_[k]=fxOpt2(cPhiF,sPhiF,cThetaF,sThetaF,ϕ,θ,ρ,Δ)

        print("check %d of %d --- step %d: %s minutes --- total time %s hours" % ( outer+1, outernum, k+1, (time.time() - inner_time)/60. , ((time.time() - outer_time)/3600) ) )

    print('loop complete. saving array as ',str(fault_name)+'_'+str(datetime.date.today())+'_precomp_fx.npy')
    np.save(str(fault_name)+'_'+str(datetime.date.today())+'_precomp_fx.pyc',fxO_)
    print('save complete')

    return fxO_
