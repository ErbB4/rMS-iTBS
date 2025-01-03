import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import matplotlib.gridspec as gridspec
import seaborn as sns 
import pingouin as pg 

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
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), fontsize=6,color='w',ha='center', va='bottom')


def data_split(data,region_name):
    region_data = data[data["region"]==region_name]
    return region_data

def get_mean_and_sem(each_group_data):
    mean = np.mean(each_group_data["norm_count_by_sham"])
    sem  = sp.stats.sem(each_group_data["norm_count_by_sham"])
    return mean,sem

def gather_for_plot(data,groupnames,region_name):
    region_data = data_split(data,region_name)
    means = []
    sems  = []
    for group in groupnames:
        each_group = region_data[region_data["group"]==group]
        mean,sem = get_mean_and_sem(each_group)
        means.append(mean)
        sems.append(sem)
    return np.array(means),np.array(sems)

def stats(data,groupnames,region_name):
    region_data = data_split(data,region_name)
    if len(groupnames)==2:
        data_1 = region_data[region_data["group"]==groupnames[0]]
        data_2 = region_data[region_data["group"]==groupnames[1]]
        p1 = sp.stats.mannwhitneyu(data_1["norm_count_by_sham"],data_2["norm_count_by_sham"])[1]

    if len(groupnames)==4:
        data_1 = region_data[region_data["group"]==groupnames[0]]
        data_2 = region_data[region_data["group"]==groupnames[1]]
        data_3 = region_data[region_data["group"]==groupnames[2]]
        data_4 = region_data[region_data["group"]==groupnames[3]]
        p1 = sp.stats.mannwhitneyu(data_1["norm_count_by_sham"],data_2["norm_count_by_sham"])[1]
        p2 = sp.stats.mannwhitneyu(data_3["norm_count_by_sham"],data_4["norm_count_by_sham"])[1]

    if len(groupnames)==6:
        data_1 = region_data[region_data["group"]==groupnames[0]]
        data_2 = region_data[region_data["group"]==groupnames[1]]
        data_3 = region_data[region_data["group"]==groupnames[2]]
        data_4 = region_data[region_data["group"]==groupnames[3]]
        data_5 = region_data[region_data["group"]==groupnames[2]]
        data_6 = region_data[region_data["group"]==groupnames[3]]

        p1 = sp.stats.mannwhitneyu(data_1["norm_count_by_sham"],data_2["norm_count_by_sham"])[1]
        p2 = sp.stats.mannwhitneyu(data_3["norm_count_by_sham"],data_4["norm_count_by_sham"])[1]
        p3 = sp.stats.mannwhitneyu(data_5["norm_count_by_sham"],data_6["norm_count_by_sham"])[1]

    return p1,p2


def plot_bar(ax,data,groupnames,region_name,xs,colors):
    means,sems = gather_for_plot(data,groupnames,region_name)
    bars = ax.bar(xs, means,yerr=sems,color=colors,ecolor='k',capsize=3.,align='center')
    ax.set_title(str(region_name))
    ax.set_xticks(xs,groupnames)
    samplesize = plot_individual_datapoint(ax,data,groupnames,region_name,xs,colors)
    autolabel(bars,samplesize,xs)
    p1,p2 = stats(data,groupnames,region_name)
    print(groupnames)
    print(region_name)
    print(p1)
    print(p2)
    
    if p1<0.005:
        ax.text(xs[1],means[1]+6*sems[1],"***",va='bottom',ha='center',fontsize=10.)
    elif p1<0.01:
        ax.text(xs[1],means[1]+3*sems[1],"**",va='bottom',ha='center',fontsize=10.)
    elif p1<0.05:
        ax.text(xs[1],means[1]+3*sems[1],"*",va='bottom',ha='center',fontsize=10.)
    elif p1>0.05:
        ax.text(xs[1],means[1]+3*sems[1],"ns",va='bottom',ha='center',fontsize=8.)
    
    if p2<0.001:
        ax.text(xs[3],means[3]+3*sems[3],"***",va='bottom',ha='center',fontsize=10.)
    elif p2<0.05:
        ax.text(xs[3],means[3]+3*sems[3],"*",va='bottom',ha='center',fontsize=10.)
    elif p2>0.05:
        ax.text(xs[3],means[1]+3*sems[1],"ns",va='bottom',ha='center',fontsize=8.)

def plot_individual_datapoint(ax,data,groupnames,region_name,xs,colors):
    region_data = data_split(data,region_name)
    samplesize = []
    for left,group_ID in enumerate(groupnames):
        each_group = region_data[region_data["group"]==group_ID]
        ax.plot(np.random.rand(len(each_group))*0.5+xs[left]-0.25, each_group["norm_count_by_sham"].values,".",color='#6c757d')
        samplesize.append(len(each_group))
    return samplesize


def plot(data,groupnames,region_name):
    fig = plt.figure(figsize=(cm2inch(4), cm2inch(3)))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=0.85,bottom=0.2,left=0.25,right=0.95,hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs1[0,0])
    xs = [1,2,4,5]
    colors= ['#293241','#98c1d9','#293241','#ee6c4d']
    plot_bar(ax1,data,groupnames,region_name,xs,colors)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(["control","1.5h","control","24h"])
    ax1.set_ylabel("c-Fos+ cell (fold)")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')



### call functions ###
data = pd.read_csv("90min_24h_cfos_count.csv")


groupnames = ["sham","rMS","sham-24h","24h"]

plot(data,groupnames,'EC')
plt.savefig("EC.svg")

plot(data,groupnames,'DG')
plt.savefig("DG.svg")

plot(data,groupnames,'CA3')
plt.savefig("CA3.svg")

plot(data,groupnames,'CA1')
plt.savefig("CA1.svg")



plt.show()