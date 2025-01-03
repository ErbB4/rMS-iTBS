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
plt.rcParams["lines.markersize"]=2.
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

def plot_ridge_plot(data,color):
    gs = (gridspec.GridSpec(len(np.unique(data["time"])),1))
    fig = plt.figure(figsize=(cm2inch(2.1), cm2inch(3.5)))
    gs.update(hspace= 0,left=0.02,right=0.98,top=0.99,bottom=0.2)
    i = 0
    ax_objs = []
    for time in np.unique(data["time"]):
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        plot = (data[data.time == time]
                .RawIntDen.plot.kde(ax=ax_objs[-1],color='none', lw=0.01)
               )
        x = plot.get_children()[0]._x
        y = plot.get_children()[0]._y
        ax_objs[-1].fill_between(x,y,color=color,alpha=np.linspace(0.1,1.,len(np.unique(data["time"])))[i])
        ax_objs[-1].axvline(x=data[data.time == time].RawIntDen.mean(),ymin=0,ymax=0.25,color="orange")
        ax_objs[-1].set_xlim(-10000, 30000)
        ax_objs[-1].set_ylim(0,0.00015)

        rect = ax_objs[-1].patch
        rect.set_alpha(0)
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')
        if i == len(np.unique(data["time"]))-1:
            ax_objs[-1].set_yticks([])
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].axis("off")
        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
        i += 1

    ax_objs[-1].set_xticks([0,15000,30000])
    ax_objs[-1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    plt.savefig(str(np.unique(data["group"])[0])+".svg")



def plot_summed_spine_size(data):
    fig = plt.figure(figsize=(cm2inch(6), cm2inch(3.7)))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=0.9,bottom=0.25,left=0.25,right=0.98,hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs1[0,0])
    xs = [0,1,2,3,4,4.6,5.1,5.6,6.1,6.6,10.1]
    colors= ['#293241','#98c1d9']
    control = data[data["group"]=="control"].groupby(["time_cor"])["summed_size"].mean()
    rMS = data[data["group"]=="rMS"].groupby(["time_cor"])["summed_size"].mean()

    control_sem = data[data["group"]=="control"].groupby(["time_cor"])["summed_size"].sem()
    rMS_sem = data[data["group"]=="rMS"].groupby(["time_cor"])["summed_size"].sem()
    ax1.plot(xs,control/control[4],'.-',color=colors[0],linewidth=1.5)
    ax1.plot(xs,rMS/rMS[4],'.-',color=colors[1],linewidth=1.5)

def plot_average_spine_size(data):
    fig = plt.figure(figsize=(cm2inch(6), cm2inch(3.7)))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=0.9,bottom=0.25,left=0.25,right=0.98,hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs1[0,0])
    xs = [0,1,2,3,4,4.6,5.1,5.6,6.1,6.6,10.1]
    colors= ['#293241','#98c1d9']
    control = data[data["group"]=="control"].groupby(["time_cor"])["normed_size"].mean().values
    rMS = data[data["group"]=="rMS"].groupby(["time_cor"])["normed_size"].mean().values

    control_sem = data[data["group"]=="control"].groupby(["time_cor"])["normed_size"].sem()
    rMS_sem = data[data["group"]=="rMS"].groupby(["time_cor"])["normed_size"].sem()
    ax1.plot(xs,control,'.-',color=colors[0],linewidth=1.,label="control")
    ax1.plot(xs,rMS,'.-',color=colors[1],linewidth=1.,label="rMS")
    ax1.fill_between(xs,control-control_sem,control+control_sem,color=colors[0],alpha=0.2)
    ax1.fill_between(xs,rMS-rMS_sem,rMS+rMS_sem,color=colors[1],alpha=0.2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['day 0','day 1','day 2','day 3','day 4','1 h','2 h','3 h','4 h','5 h','24 h'],rotation=90)
    
    ax1.set_ylabel("Average spine size\n(normalised to Day 4)")
    ax1.axvline(x=4.1,ymin=0,ymax=0.75,linestyle='--',color='#ea638c',linewidth=0.5)
    ax1.text(4.1,1.65,"600 pulses\niTBS-rMS",fontsize=5,color='#ea638c',ha="center",va="bottom")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.legend(bbox_to_anchor=(0.7, 0.75, 0.3, 0.2), loc=1,ncol=1, mode="expand", borderaxespad=0.,frameon=False)
    #ax1.text(4.6,1.55,"ns",fontsize=8)
    ax1.set_xlim(-0.5,10.5)

    plt.savefig("spinesize_normed.svg")


### call functions ###
data = pd.read_csv("spine_size_merged.csv")
plot_average_spine_size(data) #works
plot_ridge_plot(data[data["group"]=="control"],"#293241") #works
plot_ridge_plot(data[data["group"]=="rMS"],"#98c1d9") #works
plt.show()