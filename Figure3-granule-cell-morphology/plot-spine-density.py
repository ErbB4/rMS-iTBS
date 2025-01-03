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

def autolabel(bars,samplesize,xs):
    for ii,bar in enumerate(bars):
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), ha='center', va='bottom')


def normalize_spine_density(data):
    custom_dict = {'day0': 0, 'day1': 1, 'day2': 2, 'day3': 3, 'day4': 4, '1h': 4.6, '2h': 5.1, '3h': 5.6, '4h': 6.1, '5h': 6.6, '24h': 10.1}
    for group in np.unique(data["group"].values):
        groupdata = data[data["group"]==group]
        for segment in np.unique(groupdata["segment"].values):
            segmentdata = groupdata[groupdata["segment"]==segment]
            initial_value = segmentdata[segmentdata["time"]=="day4"]["spine_density"].values[0]
            data.loc[(data["group"]==group) & (data["segment"]==segment),"normed_density"] = data.loc[(data["group"]==group) & (data["segment"]==segment)]["spine_density"].values/initial_value
    data.sort_values(by=['time'], key=lambda x: x.map(custom_dict))
    data.replace("sham","control",inplace=True)

    return data


def plot_line(ax,data,xs,colors,analysis_type):
    #split group data and define color for each group
    groups = np.unique(data["group"].values)
    for group in groups:
        if group == "control":
            color = colors[0]
        else:
            color = colors[1]
        groupdata = data[data["group"]==group]
        #get each segment data from each group
        normed = []
        for segment in np.unique(groupdata["segment"].values):
            segment_data = groupdata[groupdata["segment"]==segment]
            if analysis_type == "raw_data":
                ax.plot(xs,segment_data["spine_density"].values,'-',color=color,alpha=.3)
            if analysis_type == "normed":
                ax.plot(xs,segment_data["normed_density"].values,'-',color=color,alpha=.3)
            normed.append(segment_data["normed_density"].values)
        mean = np.mean(np.array(normed),axis=0)
        sem = sp.stats.sem(np.array(normed),axis=0)

        if analysis_type == "normed":
            ax.plot(xs,mean,'.-',linewidth=1.,color=color,label=group)
            ax.fill_between(xs,mean-sem,mean+sem,color=color,alpha=0.2)


        ax.legend(loc='best')
        ax.set_ylim(0.45,1.65)


def plot(data,analysis_type):
    fig = plt.figure(figsize=(cm2inch(6), cm2inch(3.7)))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=0.9,bottom=0.25,left=0.25,right=0.98,hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs1[0,0])
    xs = [0,1,2,3,4,4.6,5.1,5.6,6.1,6.6,10.1]
    colors= ['#293241','#98c1d9']
    data = normalize_spine_density(data)
    plot_line(ax1,data,xs,colors,analysis_type)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(['day 0','day 1','day 2','day 3','day 4','1 h','2 h','3 h','4 h','5 h','24 h'],rotation=90)
    ax1.set_ylabel("Spine density\n(normalised to Day 4)")
    ax1.axvline(x=4.1,ymin=0,ymax=0.75,linestyle='--',color='#ea638c')
    ax1.text(4.1,1.45,"600 pulses\niTBS-rMS",fontsize=5,color='#ea638c',ha="center",va="bottom")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.legend(bbox_to_anchor=(0.7, 0.75, 0.3, 0.2), loc=1,ncol=1, mode="expand", borderaxespad=0.,frameon=False)
    ax1.text(4.6,1.3,"ns",fontsize=8)

    if analysis_type == "raw_data":
        plt.savefig("spinedensity_raw.svg")
    if analysis_type == "normed":
        plt.savefig("spinedensity_normed.svg")
    if analysis_type == "merged":
        plt.savefig("spinedensity_merged.svg")


def stats(data):
    dates = np.unique(data["time"])
    for date in dates:
        date_all = data[data["time"]==date]
        data_sham_sub = date_all[date_all["group"]=="control"]
        data_rMS_sub = date_all[date_all["group"]=="rMS"]
        print(date)
        p = sp.stats.mannwhitneyu(data_sham_sub["normed_density"],data_rMS_sub["normed_density"])[1]
        print(p)




### call functions ###
data = pd.read_csv("spine_density_old.csv")
data["segment"] = data["batch"].astype(str)+data["segment"]

plot(data,'normed')
plt.show()