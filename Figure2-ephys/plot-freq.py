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
def get_freq(data):
    freq_data = pd.DataFrame({"freq":[],"group":[],"cell_ID":[]})
    for group in np.unique(data["group"]):
        group_data = data[data["group"]==group]
        for cell in np.unique(group_data["cell_ID"]):
            cell_data = group_data[group_data["cell_ID"]==cell]
            intervals = cell_data["Interevent-Interval_[ms]"].values 
            ave_interval = np.mean(intervals/1000.)
            freq_data_slice = pd.DataFrame({"freq":[1./ave_interval],"group":[group],"cell_ID":[cell]})
            freq_data = pd.concat([freq_data,freq_data_slice])
    return freq_data


def plot_scatter(data,ax,colors):
    sns.boxplot(data=data,x="group",y="freq",order=["control","stimulated","stimulated_24h"],color='w',showfliers = False,saturation=1.,ax=ax)
    plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(data=data,x="group", y="freq", size=2.,order=["control","stimulated","stimulated_24h"],palette=colors,ax=ax)


def sep_data(data):
    data_ctr   = data[data["group"]=="control"]
    data_3h    = data[data["group"]=="stimulated"]
    data_24h   = data[data["group"]=="stimulated_24h"]
    return data_ctr,data_3h,data_24h

def stats(data):
    data_ctr,data_3h,data_24h = sep_data(data)
    t = sp.stats.kruskal(data_ctr["freq"].values,data_3h["freq"].values,data_24h["freq"].values)[1]
    a = sp.stats.mannwhitneyu(data_ctr["freq"].values,data_3h["freq"].values)[1]
    b = sp.stats.mannwhitneyu(data_ctr["freq"].values,data_24h["freq"].values)[1]

    print(t)
    print(a)
    print(b)

    return t,a,b

#plot freq plots
################ load interval raw data ###############
T = "./"
data_raw = pd.read_csv(T+"merged_data_intereventinterval.csv")


### call functions ###
data = get_freq(data_raw)

fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.15,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])

colors = ['#293241','#98c1d9','#ee6c4d']
plot_scatter(data,ax1,colors)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xlim(-1,3)
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(["control","2-4h","24h"])
ax1.set_ylabel("mEPSC freq. (Hz)")
ax1.set_xlabel("")

t,a,b = stats(data)
if a<0.01:
    ax1.text(1,3.9,"**",fontsize=10,ha='center',va='bottom')
if b<0.001:
    ax1.text(2,3.5,"***",fontsize=10,ha='center',va='bottom')


plt.savefig("freq.svg")

data.to_csv("merged_data_freq.csv")


plt.show()