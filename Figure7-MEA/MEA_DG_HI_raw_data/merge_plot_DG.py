import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import matplotlib.gridspec as gridspec
import seaborn as sns 
import pingouin as pg 
import statsmodels.formula.api as smf
from scipy import stats
from numpy import trapz
import random 

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

def clean_data(results):
    h0_rms = []
    h0_ctr = []

    h3_rms = []
    h3_ctr = []

    h24_rms = []
    h24_ctr = []

    for i in np.arange(0,len(results),1):
        entry = results.loc[i,:].values
        if entry[0] == "h0":
            if entry[1] == "rms":
                h0_rms.append([item for item in np.array(entry[3::])])
            if entry[1] == "ctr":
                h0_ctr.append([item for item in np.array(entry[3::])])

        if entry[0] == "h3":
            if entry[1] == "rms":
                h3_rms.append([item for item in np.array(entry[3::])])
            if entry[1] == "ctr":
                h3_ctr.append([item for item in np.array(entry[3::])])

        if entry[0] == "h24":
            if entry[1] == "rms":
                h24_rms.append([item for item in np.array(entry[3::])])
            if entry[1] == "ctr":
                h24_ctr.append([item for item in np.array(entry[3::])])
    return np.array(h0_ctr),np.array(h0_rms),np.array(h3_ctr),np.array(h3_rms),np.array(h24_ctr),np.array(h24_rms)


def plot_individual_group(subdata,color,label,ax,offset,ratio,individual=False):
    cutting = 10
    subdata = subdata[:,cutting:-1]

    mean = np.mean(subdata,axis=0)
    sem = sp.stats.sem(subdata,axis=0)
    x = np.linspace(0,600,len(h0_ctr[0]))[cutting:-1]

    if individual==False:
        ax.plot(x,ratio*(mean)+offset,color=color,label=label,linewidth=0.5)
        ax.fill_between(x,ratio*(mean+sem)+offset,ratio*(mean-sem)+offset,color=color,alpha=0.2)

    if individual==True:
        for trace in subdata:
            ax.plot(x,trace,color=color,label=label)

    ax.legend(loc='best',frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])



def autolabel(bars,samplesize,xs):
    for ii,bar in enumerate(bars):
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), fontsize=6,color='w',ha='center', va='bottom')



def get_AUC_and_compare(subdata1,subdata2,name):
    x = np.linspace(0,600,len(h0_ctr[0]))
    AUCs_1 = []
    AUCs_2 = []

    for trace in subdata1:
        AUCs_1.append(trapz(y=trace,x=x))

    for trace in subdata2:
        AUCs_2.append(trapz(y=trace,x=x))

    #print(np.array(AUCs_1))
    #print(np.array(AUCs_2))
    print(sp.stats.mannwhitneyu(np.array(AUCs_1),np.array(AUCs_2)))
    df = pd.DataFrame({"AUC1":[AUCs_1], "AUC2":[AUCs_2]})
    df.to_csv(str(name)+".csv")
    return AUCs_1,AUCs_2



def get_AUC_all_data(data_raw):
    h0_ctr,h0_rms,h3_ctr,h3_rms,h24_ctr,h24_rms = clean_data(data_raw)
    a1,a2 = get_AUC_and_compare(h0_ctr,h0_rms,"DG-h0")
    a3,a4 = get_AUC_and_compare(h3_ctr,h3_rms,"DG-h3")
    a5,a6 = get_AUC_and_compare(h24_ctr,h24_rms,"DG-h24")
    return a1,a2,a3,a4,a5,a6


def gather_for_plot(data):
    a1,a2,a3,a4,a5,a6 = get_AUC_all_data(data)

    means = []
    sems  = []
    for list_AUC in [a1,a2,a3,a4,a5,a6]:
        means.append(np.mean(list_AUC))
        sems.append(sp.stats.sem(list_AUC))
    return np.array(means),np.array(sems)


def plot_bar(data,ax,colors,xs):
    means,sems = gather_for_plot(data)
    bars = ax.bar(xs, means,yerr=sems,color=colors,ecolor='k',capsize=3.,align='center')
    samplesize = plot_individual_datapoint(data,ax,colors,xs)
    autolabel(bars,samplesize,xs)
    ax.set_xticks(xs)
    ax.set_xticklabels(["ctr","0h","ctr","3h","ctr","24h"])
    ax.set_ylabel("AUC")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.text(xs[1],means[1]+6*sems[1],"ns",va='bottom',ha='center',fontsize=8.)
    ax.text(xs[3],means[3]+6*sems[3],"ns",va='bottom',ha='center',fontsize=8.)
    ax.text(xs[5],means[5]+6*sems[5],"ns",va='bottom',ha='center',fontsize=8.)

def plot_individual_datapoint(data,ax,colors,xs):
    a1,a2,a3,a4,a5,a6 = get_AUC_all_data(data)
    samplesize = []
    for idx,list_AUC in enumerate([a1,a2,a3,a4,a5,a6]):
        ax.plot(np.random.rand(len(list_AUC))*0.5+xs[idx]-0.25, list_AUC,".",color='#6c757d')
        samplesize.append(len(list_AUC))
    return np.array(samplesize)



def using_permutation_to_compare(subdata1,subdata2):
    sample_size_1 = len(subdata1)
    sample_size_2 = len(subdata2)

    trace_pool = np.vstack((subdata1,subdata2))
    diffs_pool = []

    original_diff = np.sum(np.mean(subdata1,axis=0) - np.mean(subdata2,axis=0))


    for i in np.arange(0,5000,1):
        ids_1 = random.sample(range(sample_size_1+sample_size_2),sample_size_1)
        ids_2 = [item for item in np.arange(0,sample_size_1+sample_size_2,1) if item not in ids_1]
        new_sample_1 = trace_pool[ids_1]
        new_sample_2 = trace_pool[ids_2]

        diff = np.sum(np.mean(new_sample_1,axis=0) - np.mean(new_sample_2,axis=0))
        diffs_pool.append(diff)
    return original_diff, diffs_pool



    AUCs_1 = []
    AUCs_2 = []

    for trace in subdata1:
        AUCs_1.append(trapz(y=trace,x=x))

    for trace in subdata2:
        AUCs_2.append(trapz(y=trace,x=x))

    #print(np.array(AUCs_1))
    #print(np.array(AUCs_2))
    #print(sp.stats.mannwhitneyu(np.array(AUCs_1),np.array(AUCs_2)))



############## plot average traces ###################

fig = plt.figure(1,figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.99,bottom=0.01,left=0.05,right=0.95,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])

ax1.plot([120,240],[1.,1.],color='k',linewidth=1.)
ax1.plot([120,120],[1.,1.5],color='k',linewidth=1.)

data_raw = pd.read_csv("DG-MEA.csv")
h0_ctr,h0_rms,h3_ctr,h3_rms,h24_ctr,h24_rms = clean_data(data_raw)

plot_individual_group(h0_ctr,"#293241","control",ax1,offset=0,ratio=1.)
plot_individual_group(h0_rms,"green","immediately after rMS",ax1,offset=-0.2,ratio=-1.)


plot_individual_group(h3_ctr,"#293241","control",ax1,offset=-1.5,ratio=1.)
plot_individual_group(h3_rms,"#98c1d9","3h-post rMS",ax1,offset=-1.7,ratio=-1.)

plot_individual_group(h24_ctr,"#293241","control",ax1,offset=-4.,ratio=1.)
plot_individual_group(h24_rms,"#ee6c4d","24h-post rMS",ax1,offset=-4.2,ratio=-1.)

plt.figure(1)
plt.savefig("MEA_average_FR_DG.svg")



##################### plot stats for AUCs ###################
fig = plt.figure(2,figsize=(cm2inch(5.), cm2inch(3)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.99,bottom=0.01,left=0.05,right=0.95,hspace=0.1,wspace=0.1)
ax2 = plt.subplot(gs1[0,0])

colors = ["#293241","green","#293241","#98c1d9","#293241","#ee6c4d"]
xs = [1,2,4,5,7,8]
plot_bar(data_raw,ax2,colors,xs)
plt.figure(2)
plt.savefig("MEA_AUC_DG.svg")

plt.show()