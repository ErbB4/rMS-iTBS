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


fig = plt.figure(figsize=(cm2inch(2.2), cm2inch(1.2)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.05,left=0.05,right=.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])


T = "./raw_data/control/"
trace = pyabf.ABF(T+"2022_03_10_0011.abf")
trace.setSweep(sweepNumber=0,channel=2)

ax1.plot(trace.sweepX*60,trace.sweepY-26,color='k',label='control')
ax1.set_ylim(-51,20)
ax1.set_xlim(6202.5,6204.5)
ax1.axis('off')

plt.savefig("event.svg")

plt.show()