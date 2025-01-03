import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import matplotlib.gridspec as gridspec
import seaborn as sns 
import pingouin as pg 
import statsmodels.formula.api as smf

####figure settings####
plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["axes.linewidth"]=.5
plt.rcParams["lines.linewidth"]=1.
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

def autolabel(bars,samplesize,xs):
    for ii,bar in enumerate(bars):
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), ha='center', va='bottom')




################ functions ##########

def sep_data(data):
    data_ctr   = data[data["group"]=="control"]
    data_3h    = data[data["group"]=="stimulated"]
    data_24h   = data[data["group"]=="stimulated_24h"]
    return data_ctr,data_3h,data_24h


def get_hists(subdata,parameter="Peak_Amplitude_[pA]"):
    hists = []
    for cell_name in np.unique(subdata["cell_ID"]):
        celldata  = subdata[subdata["cell_ID"]==cell_name]
        cell_values = celldata[parameter]
        cell_values = abs(cell_values)
        hist = np.histogram(cell_values,bins=np.arange(0,150,10),density=False)
        hists.append(hist[0])
    return np.array(hists),hist[1]

def plot_hists_amps(subdata,ax):
    
    hists,lefts = get_hists(subdata)

    mean = np.mean(hists,axis=0)
    sem  = sp.stats.sem(hists,axis=0)

    if np.unique(subdata["group"])=="control":
        ax.plot(lefts[0:-1],mean,'-',color='#293241',label='control')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#293241',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated":
        ax.plot(lefts[0:-1],mean,'-',color='#98c1d9',label='2-4h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#98c1d9',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated_24h":
        ax.plot(lefts[0:-1],mean,'-',color='#ee6c4d',label='24h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#ee6c4d',alpha=0.1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(lefts[::5])

    ax.set_xlabel("mEPSC amplitude (pA)")
    ax.set_ylabel("count")
    ax.legend(loc='upper right',frameon=False)



durations_ctr = pd.read_csv("./raw_data/control/duration.csv")
durations_h3 = pd.read_csv("./raw_data/stimulated/duration.csv")
durations_h24 = pd.read_csv("./raw_data/stimulated_24h/duration.csv")

def get_hists_normed_by_duration(subdata,duration,parameter="Peak_Amplitude_[pA]"):
    hists = []
    for cell_name in np.unique(subdata["cell_ID"]):
        celldata  = subdata[subdata["cell_ID"]==cell_name]
        cell_values = celldata[parameter]
        cell_values = abs(cell_values)
        hist = np.histogram(cell_values,bins=np.arange(0,150,10),density=False)
        hists.append(hist[0])
    return np.array(hists),hist[1]





def plot_hists_amps_normed(subdata,ax):
    
    hists,lefts = get_hists(subdata)

    mean = np.mean(hists,axis=0)
    sem  = sp.stats.sem(hists,axis=0)

    if np.unique(subdata["group"])=="control":
        ax.plot(lefts[0:-1],mean,'-',color='#293241',label='control')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#293241',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated":
        ax.plot(lefts[0:-1],mean,'-',color='#98c1d9',label='2-4h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#98c1d9',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated_24h":
        ax.plot(lefts[0:-1],mean,'-',color='#ee6c4d',label='24h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#ee6c4d',alpha=0.1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(lefts[::5])

    ax.set_xlabel("mEPSC amplitude (pA)")
    ax.set_ylabel("count")
    ax.legend(loc='upper right',frameon=False)


def get_ave_peak_amp(data):
    amp_data = pd.DataFrame({"ave_amp":[],"group":[],"cell_ID":[],"sem_amp":[]})
    for group in np.unique(data["group"]):
        group_data = data[data["group"]==group]
        for cell in np.unique(group_data["cell_ID"]):
            cell_data = group_data[group_data["cell_ID"]==cell]
            amps = -1.*cell_data["Peak_Amplitude_[pA]"].values 
            ave_amp = np.mean(amps)
            sem_amp = sp.stats.sem(amps)
            amp_data_slice = pd.DataFrame({"ave_amp":[ave_amp],"group":[group],"cell_ID":[cell],"sem_amp":[sem_amp]})
            amp_data = pd.concat([amp_data,amp_data_slice])
    return amp_data



def plot_scatter(data,ax,colors):
    sns.boxplot(data=data,x="group",y="ave_amp",order=["control","stimulated","stimulated_24h"],color='w',showfliers = False,saturation=1.,ax=ax)
    plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(data=data,x="group", y="ave_amp", size=2.,order=["control","stimulated","stimulated_24h"],palette=colors,ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(-1,3)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["control","2-4h","24h"])
    ax.set_ylabel("mEPSC ampulitude (pA)")
    ax.set_xlabel("")


################ load amps raw data ###############
T = "./"
data = pd.read_csv(T+"merged_data_amplitude.csv")

### call functions to plot pdf ###
data_ctr,data_3h,data_24h = sep_data(data)

fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.3,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])

plot_hists_amps(data_ctr,ax1)
plot_hists_amps(data_3h,ax1)
plot_hists_amps(data_24h,ax1)
ax1.text(100,20,"***",fontsize=10,va='bottom',ha='center')
ax1.text(80,20,"ns",fontsize=8,va='bottom',ha='center')



plt.savefig("amps-pdf.svg")

### call functions to plot raw data in bars ###
fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.3,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])
colors = ['#293241','#98c1d9','#ee6c4d']
amp_data = get_ave_peak_amp(data)
plot_scatter(amp_data,ax1,colors)
ax1.text(1,35,"***",fontsize=10,va='bottom',ha='center')
ax1.text(2,35,"ns",fontsize=8,va='bottom',ha='center')


plt.savefig("amps-scatter.svg")
plt.show()


############ linear mixed model

import statsmodels.api as sm
import statsmodels.formula.api as smf

data_3h = pd.concat((data_ctr,data_3h))
data_24h = pd.concat((data_ctr,data_24h))


data_3h = data_3h.rename(columns=({'Peak_Amplitude_[pA]':'amplitude'}))
data_24h = data_24h.rename(columns=({'Peak_Amplitude_[pA]':'amplitude'}))


md1 = smf.mixedlm("amplitude ~ group", data_3h, groups=data_3h["cell_ID"])
mdf1 = md1.fit()

md2 = smf.mixedlm("amplitude ~ group", data_24h, groups=data_24h["cell_ID"])
mdf2 = md2.fit()

print(mdf1.summary(alpha=0.0001)) #still significant
print(mdf2.summary()) 
