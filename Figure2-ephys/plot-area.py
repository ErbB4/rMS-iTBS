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
def get_mean_and_sum_for_each_neuron(data):
    area_data = pd.DataFrame({"ave_area":[],"sum_area":[],"group":[],"cell_ID":[]})
    for group in np.unique(data["group"]):
        group_data = data[data["group"]==group]
        print(group)
        if group == "control":
            durations = pd.read_csv("./raw_data/control/duration.csv")
        if group == "stimulated":
            durations = pd.read_csv("./raw_data/stimulated/duration.csv")
        if group == "stimulated_24h":
            durations = pd.read_csv("./raw_data/stimulated_24h/duration.csv")
        for idx, cell in enumerate(np.unique(group_data["cell_ID"])):
            print(idx,cell)
            duration = durations["duration"][idx]
            cell_data = group_data[group_data["cell_ID"]==cell]
            areas = cell_data["Area_[pA*ms]"].values 
            ave_area = np.mean(abs(areas))
            sum_area = np.sum(abs(areas))/duration*120
            area_data_slice = pd.DataFrame({"ave_area":[ave_area],"sum_area":[sum_area],"group":[group],"cell_ID":[cell]})
            area_data = pd.concat([area_data,area_data_slice])
    return area_data


def sep_data(data):
    data_ctr   = data[data["group"]=="control"]
    data_3h    = data[data["group"]=="stimulated"]
    data_24h   = data[data["group"]=="stimulated_24h"]
    return data_ctr,data_3h,data_24h


def get_hists(subdata,parameter="Area_[pA*ms]"):
    hists = []
    for cell_name in np.unique(subdata["cell_ID"]):
        celldata  = subdata[subdata["cell_ID"]==cell_name]
        cell_values = celldata[parameter]
        cell_values = abs(cell_values)
        hist = np.histogram(cell_values,bins=np.arange(0,200,10),density=False)
        hists.append(hist[0])
    return np.array(hists),hist[1]

def plot_hists(subdata,ax):
    hists,lefts = get_hists(subdata)

    mean = np.mean(hists,axis=0)
    sem  = sp.stats.sem(hists,axis=0)

    if np.unique(subdata["group"])=="control":
        ax.plot(lefts[0:-1],mean,'-',linewidth=1.,color='#293241',label='control')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#293241',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated":
        ax.plot(lefts[0:-1],mean,'-',linewidth=1.,color='#98c1d9',label='2-4h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#98c1d9',alpha=0.1)

    if np.unique(subdata["group"])=="stimulated_24h":
        ax.plot(lefts[0:-1],mean,'-',linewidth=1.,color='#ee6c4d',label='24h')
        ax.fill_between(lefts[0:-1],mean-sem,mean+sem,color='#ee6c4d',alpha=0.1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(lefts[::5])
    ax.legend(loc='upper right',frameon=False)
    ax.set_xlabel(r"$\mathrm{mEPSC\ area\ (pA}\times\mathrm{ms})$")
    ax.set_ylabel("count")



def plot_scatter(data,ax,param,colors):
    sns.boxplot(data=data,x="group",y=param,order=["control","stimulated","stimulated_24h"],color='w',showfliers = False,saturation=1.,ax=ax)
    plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(data=data,x="group", y=param, size=2.,order=["control","stimulated","stimulated_24h"],palette=colors,ax=ax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yscale('log')

    ax.set_xlim(-1,3)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["control","2-4h","24h"])
    ax.set_ylabel(r"$\sum\ \mathrm{mEPSC\ area}$ (in 2 min)")
    ax.set_xlabel("")



def plot_scatter_ave(data,ax,colors):
    sns.boxplot(data=data,x="group",y="ave_area",order=["control","stimulated","stimulated_24h"],color='w',showfliers = False,saturation=1.,ax=ax)
    plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(data=data,x="group", y="ave_area", size=2.,order=["control","stimulated","stimulated_24h"],palette=colors,ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(-1,3)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["control","2-4h","24h"])
    ax.set_ylabel(r"$\mathrm{mEPSC\ area\ (pA}\times\mathrm{ms})$")
    ax.set_xlabel("")


################ load amps raw data ###############
T = "./"
data_raw = pd.read_csv(T+"merged_data_area.csv")

data = get_mean_and_sum_for_each_neuron(data_raw)
data.to_csv("merged_data_total_area.csv")

data_ctr,data_3h,data_24h = sep_data(data_raw)

### call functions ###
colors = ['#293241','#98c1d9','#ee6c4d']

### plot raw area ###
fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.3,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])

plot_hists(data_ctr,ax1)
plot_hists(data_3h,ax1)
plot_hists(data_24h,ax1)
ax1.text(150,20,"***",fontsize=10,va='bottom',ha='center')
ax1.text(100,20,"ns",fontsize=8,va='bottom',ha='center')

plt.savefig("area.svg")

### plot summed area ###
fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs2 = gridspec.GridSpec(1, 1)
gs2.update(top=0.95,bottom=0.15,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax2 = plt.subplot(gs2[0,0])

plot_scatter(data,ax2,"sum_area",colors)
ax2.text(1,33000,"**",fontsize=10,ha='center',va='bottom')
ax2.text(2,35000,"***",fontsize=10,ha='center',va='bottom')

plt.savefig("totla_area.svg")




### call functions to plot raw data in bars ###
fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.3,left=0.22,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])
colors = ['#293241','#98c1d9','#ee6c4d']
plot_scatter_ave(data,ax1,colors)
ax1.text(1,130,"***",fontsize=10,va='bottom',ha='center')
ax1.text(2,140,"ns",fontsize=8,va='bottom',ha='center')


plt.savefig("area-ave-scatter.svg")
plt.show()


import statsmodels.api as sm
import statsmodels.formula.api as smf

data_3h = pd.concat((data_ctr,data_3h))
data_24h = pd.concat((data_ctr,data_24h))


data_3h = data_3h.rename(columns=({'Area_[pA*ms]':'area'}))
data_24h = data_24h.rename(columns=({'Area_[pA*ms]':'area'}))


md1 = smf.mixedlm("area ~ group", data_3h, groups=data_3h["cell_ID"])
mdf1 = md1.fit()

md2 = smf.mixedlm("area ~ group", data_24h, groups=data_24h["cell_ID"])
mdf2 = md2.fit()

print(mdf1.summary(alpha=0.0001)) #still significant
print(mdf2.summary())