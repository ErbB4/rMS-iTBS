import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.gridspec as gridspec
import pingouin as pg 
from scipy import stats
import seaborn as sns

#define plotting parameters
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

data_raw = pd.read_csv("./spine_size_merged.csv")
data = data_raw.copy()


def plot_diffs_histogram_by_size(data,ax,groupname,t1,t2):
    groupdata = data[data["group"]==groupname]
    pre  = groupdata[groupdata["time"]==t1]
    post = groupdata[groupdata["time"]==t2]
    diffs = post["RawIntDen"].values - pre["RawIntDen"].values
    diffs_normed = diffs/pre["RawIntDen"].values

    new_data = pre.loc[:,["batch","culture","group","time","spine_ID","RawIntDen"]]
    new_data["size_diff"] = diffs
    new_data["size_diff_normed"] = diffs_normed

    counts,lefts = np.histogram(new_data["RawIntDen"].values,bins=8,range=(0,40000))
    sorted_data = new_data.sort_values(by="RawIntDen")

    means = []
    sems = []

    idx_manager = 0
    for i in counts:
        diffs_list = sorted_data["size_diff_normed"].values 
        means.append(np.mean(diffs_list[idx_manager:idx_manager+i]))
        sems.append(sp.stats.sem(diffs_list[idx_manager:idx_manager+i]))
        idx_manager += i
    means = np.array(means)
    sems  = np.array(sems)
    if groupname == "control":
        ax.errorbar(lefts[1::],means,yerr=sems,color='#293241',capsize=3.,label='control')
    if groupname == "rMS":
        ax.errorbar(lefts[1::],means,yerr=sems,color='#98c1d9',capsize=3.,label='rMS')

    ax.set_xticks(lefts[1::2])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.axhline(y=0,linestyle='--',color='grey',linewidth=0.5)
    ax.legend(loc="best",frameon=False)
    ax.set_ylim(-1.,1.75)


def get_2_hists(data,t1,t2):
    fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3.9)))
    gs1 = gridspec.GridSpec(1,1)
    gs1.update(top=0.99,bottom=0.2,left=0.2,right=0.99,hspace=0.05,wspace=0.15)
    ax1 = plt.subplot(gs1[0,0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plot_diffs_histogram_by_size(data,ax1,"control",t1,t2)
    plot_diffs_histogram_by_size(data,ax1,"rMS",t1,t2)
    ax1.set_xlabel("Spine size at Day 4 (a.u.)")
    ax1.set_ylabel(r"$\Delta$ spine size (normalised) at 24h")
    plt.savefig(str(t1)+"_"+str(t2)+"_spine_size_grouped.svg")

get_2_hists(data,"day4","day5")




def get_2_hists_v2(data,t1,t2):
    fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3.9)))
    gs1 = gridspec.GridSpec(1,1)
    gs1.update(top=0.99,bottom=0.2,left=0.2,right=0.99,hspace=0.05,wspace=0.15)
    ax1 = plt.subplot(gs1[0,0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plot_diffs_histogram_by_size(data,ax1,"control",t1,t2)
    plot_diffs_histogram_by_size(data,ax1,"rMS",t1,t2)
    ax1.set_xlabel("Spine size at Day 4 (a.u.)")
    ax1.set_ylabel(r"$\Delta$ spine size (normalised) at 2h")
    plt.savefig(str(t1)+"_"+str(t2)+"_spine_size_grouped.svg")

def get_2_hists_v3(data,t1,t2):
    fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3.9)))
    gs1 = gridspec.GridSpec(1,1)
    gs1.update(top=0.99,bottom=0.2,left=0.2,right=0.99,hspace=0.05,wspace=0.15)
    ax1 = plt.subplot(gs1[0,0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plot_diffs_histogram_by_size(data,ax1,"control",t1,t2)
    plot_diffs_histogram_by_size(data,ax1,"rMS",t1,t2)
    ax1.set_xlabel("Spine size at Day 4 (a.u.)")
    ax1.set_ylabel(r"$\Delta$ spine size (normalised) at 5h")
    plt.savefig(str(t1)+"_"+str(t2)+"_spine_size_grouped.svg")


def get_2_hists_v4(data,t1,t2):
    fig = plt.figure(figsize=(cm2inch(5.), cm2inch(3.9)))
    gs1 = gridspec.GridSpec(1,1)
    gs1.update(top=0.99,bottom=0.2,left=0.2,right=0.99,hspace=0.05,wspace=0.15)
    ax1 = plt.subplot(gs1[0,0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plot_diffs_histogram_by_size(data,ax1,"control",t1,t2)
    plot_diffs_histogram_by_size(data,ax1,"rMS",t1,t2)
    ax1.set_xlabel("Spine size at 2h (a.u.)")
    ax1.set_ylabel(r"$\Delta$ spine size (normalised) at 24h")
    plt.savefig(str(t1)+"_"+str(t2)+"_spine_size_grouped.svg")

get_2_hists_v2(data,"day4","day4-h2")
get_2_hists_v3(data,"day4","day4-h5")
get_2_hists_v4(data,"day4-h2","day5")


plt.show()