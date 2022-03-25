import numpy as np
import pygmt

#Modify-----------
fname = 'test_'
#-----------------

mcs_pts = np.load(fname+'_mcs_pts.npy')
mcs_cmap = np.load(fname+'_mcs_cmap.npy')

ics_pts = np.load(fname+'_ics_pts.npy')
ics_cmap = np.load(fname+'_ics_cmap.npy')

lcs_pts = np.load(fname+'_lcs_pts.npy')
lcs_cmap = np.load(fname+'_lcs_cmap.npy')

def lower_hemisphere_plot(points, cmap, name):
    theta = np.linspace(0, 2 * np.pi, 720)
    x_ = np.sqrt(2) * np.cos(theta)
    y_ = np.sqrt(2) * np.sin(theta)
    region = [-1.45, 1.45, -1.45, 1.45]

    maxfreq = max(cmap)
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="X8i", frame=True)
    fig.coast(land="#ECECEC", water="#ECECEC")
    fig.plot(x=x_, y=y_)
    pygmt.makecpt(cmap='./Plots/CBR_Wet.cpt', series=[0, 1], reverse=True)
    fig.plot(
        x=points[:, 0],
        y=points[:, 1],
        style="c0.27c",
        color=cmap / maxfreq,
        cmap=True)
    fig.contour(x=points[:, 0], y=points[:, 1], z=cmap / maxfreq, levels=0.2, pen=True)
    fig.colorbar(frame='af')
    fig.savefig('Plots/' + name + 'llhplot.pdf', dpi=720)
    return fig.show()

lower_hemisphere_plot(mcs_pts,mcs_cmap,fname+'mcs_')
lower_hemisphere_plot(ics_pts,ics_cmap,fname+'ics_')
lower_hemisphere_plot(lcs_pts,lcs_cmap,fname+'lcs_')
