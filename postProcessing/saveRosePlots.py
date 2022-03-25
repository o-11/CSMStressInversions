import numpy as np
import h5py
import plotly as plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#For plotting prediction of rake as a rose plot with observed rake overlay

#Modify--------------
exportFileName = '' #ex: 'rw_010121_0600_export.h5'
faultsFileName = 'testFaultGeometries.npy'
rakeStdFileName = 'testRakeStdDev.npy'
#End-----------------

rakePredFile = h5py.File('../stressPosteriors/'+exportFileName,'r')
rakePreds = rakePredFile['all_predicted_rakes'][:]
faultsGeom = np.load('../faultGeometryFiles/'+faultsFileName,allow_pickle=True)
rakeUnc = np.load('../uncertaintyFiles/'+rakeStdFileName)

#Functions----

def gauss_distr(x,mean,sd):
    return (1/(sd*(np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mean)/sd)**2))

def rake_roseplot(rw_rake_pred, faults_, rake_std, name='default'):

    faults = faults_[0];
    fault_num=len(faults);
    fault_pot = [np.round(faults_[1][i]/np.sum(faults_[1]),decimals=2) for i in range(fault_num)];

    ts = [[{'type': 'polar'}]]*(fault_num+1)
    fig = make_subplots(rows=fault_num+1,cols=1,specs=ts,subplot_titles=['fault: '+str(i)+', %tot_pot: '+str(fault_pot[i]) for i in range(fault_num)]);

    for i in range(fault_num):
        faultpred = np.unique(np.round(rw_rake_pred[:,i]*180/np.pi),return_counts=True)
        obs_mean=faults[i,2];
        obs_sd = rake_std[i];
        if obs_mean<0.0:
            obs_mean+=2*np.pi
        if obs_mean<np.pi/2:
            x_=np.linspace(-np.pi,np.pi,400);
            gaussian_ = gauss_distr(x_,obs_mean,obs_sd);
        elif obs_mean>3*np.pi/2:
            x_=np.linspace(np.pi,3*np.pi,400);
            gaussian_ = gauss_distr(x_,obs_mean,obs_sd);
        else:
            x_=np.linspace(0,2*np.pi,400);
            gaussian_ = gauss_distr(x_,obs_mean,obs_sd);

        max_faultpred=np.max(faultpred[1]);
        max_gaussian=np.max(gaussian_);

        fig.add_trace(go.Barpolar(
            r=[(faultpred[1]).tolist()][0]/max_faultpred,
            theta=[faultpred[0].tolist()][0]),
            row=i+1,col=1)
        fig.append_trace(go.Scatterpolar(
            r=gaussian_/max_gaussian,
            theta=x_*180/np.pi,
            marker_color='black',
            showlegend=False),
            row=i+1, col=1)
        fig.update_traces(
            marker_colorscale=plotly.colors.cyclical.Twilight,
            marker_line_color=plotly.colors.cyclical.Twilight,
            marker_line_width=0.1)

    fig.update_layout(height=800*fault_num,width=800)

    plotly.io.write_image(fig,"Plots/"+str(name)+"_rake_roseplot.png")
    plotly.offline.plot(fig, filename='Plots'+str(name)+'.html', image= 'png')
    return print('rose plots of rake predictions for '+str(fault_num)+' faults saved.')

#--------

rake_roseplot(rakePreds,faultsGeom,rakeUnc)
