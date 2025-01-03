import numpy as np 
import pyabf
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import seaborn as sns 
from scipy import stats

####figure settings####
plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["axes.linewidth"]=.5
plt.rcParams["lines.linewidth"]=.5
plt.rcParams["lines.markersize"]=2
plt.rcParams["xtick.labelsize"]=6
plt.rcParams["ytick.labelsize"]=6
plt.rcParams["font.family"] = "arial"
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams["legend.fontsize"] = 6
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

def cm2inch(value):
    return value/2.54


fig = plt.figure(figsize=(cm2inch(7), cm2inch(7.)))
gs1 = gridspec.GridSpec(6, 1)
gs1.update(top=0.95,bottom=0.05,left=0.05,right=.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])
ax1_2 = plt.subplot(gs1[1,0])

ax2 = plt.subplot(gs1[2,0])
ax2_2 = plt.subplot(gs1[3,0])

ax3 = plt.subplot(gs1[4,0])
ax3_2 = plt.subplot(gs1[5,0])

def plot_trace_centered(trace,starting_time,ax,color,label):
    amps = trace.sweepY - np.median(trace.sweepY)
    ax.plot(trace.sweepX*1000,amps,color=color,label=label)
    ax.set_ylim(-61,30)
    ax.set_xlim(starting_time,starting_time+2000.)


#control
T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/control/"
trace = pyabf.ABF(T+"2022_03_10_0003.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,35500,ax1,'k','control')

T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/control/"
trace = pyabf.ABF(T+"2022_03_31_0003.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,38500,ax1_2,'k','control')


#stimulated
T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/stimulated/"
trace = pyabf.ABF(T+"2022_03_10_0017.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,3730,ax2,'k','2-4h')

T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/stimulated/"
trace = pyabf.ABF(T+"2022_03_31_0011.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,24000,ax2_2,'k','2-4h')


#stimulated 24h
T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/stimulated_24h/"
trace = pyabf.ABF(T+"2022_03_31_0020.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,15000,ax3,'k','24h')

T = "D://!!!rMS-iTBS/Figures/Figure2-ephys/raw_data/stimulated_24h/"
trace = pyabf.ABF(T+"2022_03_31_0031.abf")
trace.setSweep(sweepNumber=0,channel=2)
plot_trace_centered(trace,28400,ax3_2,'k','24h')


ax3_2.plot([28500,28700],[-55,-55],'k-',linewidth=1.)
ax3_2.plot([28500,28500],[-55,-5],'k-',linewidth=1.)


for ax in [ax1,ax1_2,ax2,ax2_2,ax3,ax3_2]:
    ax.axis("off")
plt.savefig("trace.svg")

plt.show()