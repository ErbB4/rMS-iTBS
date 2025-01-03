import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.gridspec as gridspec
import seaborn as sns 
from scipy import stats

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

def get_means_individual(subdata):
#################### get means for plot ##############
    means = np.zeros([3,len(np.unique(subdata.culture))])
    for idx,culture in enumerate(np.unique(subdata["culture"])):
        culture_data = subdata[subdata["culture"]==culture]
        time_1 = culture_data.loc[culture_data["timing"]=="pre"].coef
        time_2 = culture_data.loc[culture_data["timing"]=="3h-post"].coef
        time_3 = culture_data.loc[culture_data["timing"]=="24h-post"].coef

        means[0,idx] = np.mean(time_1)
        means[1,idx] = np.mean(time_2)
        means[2,idx] = np.mean(time_3)
    return means

def get_means_individual_two_time_points(subdata):
#################### get means for plot ##############
    means = np.zeros([2,len(np.unique(subdata.culture))])
    for idx,culture in enumerate(np.unique(subdata["culture"])):
        culture_data = subdata[subdata["culture"]==culture]
        time_2 = culture_data.loc[culture_data["timing"]=="3h-post"].coef
        time_3 = culture_data.loc[culture_data["timing"]=="24h-post"].coef

        means[0,idx] = np.mean(time_2)
        means[1,idx] = np.mean(time_3)
    return means

def get_means_individual_two_groups(batchdata):
#################### get means for two groups ##############
    control_data = batchdata[batchdata["group"]=="control"]
    rMS_data = batchdata[batchdata["group"]=="rMS"]
    if len(np.unique(control_data.timing))==3:
        means_control = get_means_individual(control_data)
        means_rMS = get_means_individual(rMS_data)
    if len(np.unique(control_data.timing))==2:
        means_control = get_means_individual_two_time_points(control_data)
        means_rMS = get_means_individual_two_time_points(rMS_data)
    return np.mean(means_control,axis=1),np.mean(means_rMS,axis=1)
    
def get_means_individual_two_groups_pooled(batchdata):
#################### get means for two groups ##############
    control_data = batchdata[batchdata["group"]=="control"]
    rMS_data = batchdata[batchdata["group"]=="rMS"]

    means_control = np.array([np.mean(np.unique(control_data[control_data["timing"]=="3h-post"].coef)), np.mean(np.unique(control_data[control_data["timing"]=="24h-post"].coef))])
    means_rMS = np.array([np.mean(np.unique(rMS_data[rMS_data["timing"]=="3h-post"].coef)),np.mean(np.unique(rMS_data[rMS_data["timing"]=="24h-post"].coef))])
    sems_control = np.array([sp.stats.sem(np.unique(control_data[control_data["timing"]=="3h-post"].coef)),sp.stats.sem(np.unique(control_data[control_data["timing"]=="24h-post"].coef))])
    sems_rMS = np.array([sp.stats.sem(np.unique(rMS_data[rMS_data["timing"]=="3h-post"].coef)),sp.stats.sem(np.unique(rMS_data[rMS_data["timing"]=="24h-post"].coef))])

    return np.array([means_control[0],means_rMS[0],means_control[1],means_rMS[1]]),np.array([sems_control[0],sems_rMS[0],sems_control[1],sems_rMS[1]])

def stats_pooled(batchdata):
#################### stats ##############
    control_data = batchdata[batchdata["group"]=="control"]
    rMS_data = batchdata[batchdata["group"]=="rMS"]
    print("3h-stats")
    print(sp.stats.mannwhitneyu(np.unique(control_data[control_data["timing"]=="3h-post"].coef),np.unique(rMS_data[rMS_data["timing"]=="3h-post"].coef)))
    print("24h-stats")
    print(sp.stats.mannwhitneyu(np.unique(control_data[control_data["timing"]=="24h-post"].coef),np.unique(rMS_data[rMS_data["timing"]=="24h-post"].coef)))


def plot_line_plot(batchdata,ax,pool=True):
#################### plot two groups ##############
    if pool == False:
        means_control, means_rMS = get_means_individual_two_groups(batchdata)
    if pool == True:
        means_control, means_rMS = get_means_individual_two_groups_pooled(batchdata)

    if means_control.shape[0]==3:
        ax.plot([0,1,2],means_control,color='k')
        ax.plot([0,1,2],means_rMS,color='r')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(["pre-","3h-post","24h-post"])

    if means_control.shape[0]==2:
        ax.plot([1,2],means_control,color='k')
        ax.plot([1,2],means_rMS,color='r')
        ax.set_xticks([1,2])
        ax.set_xticklabels(["3h-post","24h-post"])




################ functions for plotting ###########
def autolabel(bars,samplesize,xs):
    for ii,bar in enumerate(bars):
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), fontsize=6,color='w',ha='center', va='bottom')

def autolabel_culture(bars,samplesize,xs):
    for ii,bar in enumerate(bars):
        plt.text(xs[ii], 0.0, '%s'% (samplesize[ii]), fontsize=6,color='w',ha='center', va='bottom')

def gather_for_plot(data):
    means = []
    sems  = []
    for timing in ["3h-post","24h-post"]:
        timingdata = data[data["timing"]==timing]
        for group in ["control","rMS"]:
            each_group = timingdata[timingdata["group"]==group]
            mean,sem = get_mean_and_sem(each_group)
            means.append(mean)
            sems.append(sem)
    return np.array(means),np.array(sems)

def plot_bar(ax,data,region_name):
    means,sems = get_means_individual_two_groups_pooled(data)
    colors = ['#293241','#98c1d9','#293241','#ee6c4d']
    xs = [1,2,4,5]
    bars = ax.bar(xs, means,yerr=sems,color=colors,ecolor='k',capsize=3.,align='center')
    ax.set_title(str(region_name))
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["control","3h","control","24h"])
    ax.set_ylabel(r"$cf$")
    samplesize_culture,samplesize_neuron = plot_individual_datapoint(ax,data,xs,colors)

    ax.text(2,means[1]+2*sems[1],"*",fontsize=10,ha='center',va='bottom')
    ax.text(5,means[3]+2*sems[3],"*",fontsize=10,ha='center',va='bottom')
    ax.set_ylim(0,means[3]+4*sems[3])

    #autolabel(bars,samplesize_neuron,xs)
    autolabel_culture(bars,samplesize_culture,xs)

def plot_individual_datapoint(ax,data,xs,colors):
    i=0
    samplesize_culture = []
    samplesize_neuron = []

    for timing in ["3h-post","24h-post"]:
        timingdata = data[data["timing"]==timing]
        for group in ["control","rMS"]:
            each_group = timingdata[timingdata["group"]==group]
            #ax.plot(np.random.rand(len(each_group))*0.5+xs[i]-0.25, each_group["firing_rate"]/120.,".",color='#6c757d')
            samplesize_culture.append(len(np.unique(each_group.culture)))
            samplesize_neuron.append(len(each_group))
            i += 1
    return samplesize_culture,samplesize_neuron

#################### load data and plot ##############
data = pd.read_csv("calcium_data_hilus.csv")
data =data.dropna()

#plot only 3h-post and 24h-post
fig = plt.figure(figsize=(cm2inch(4.3), cm2inch(3.7)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.9,bottom=0.25,left=0.1,right=0.98,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[0,0])

data_update = data[data["prep_date"]!="15-03-2022"]
data_update.replace("24h-from-3h","24h-post",inplace=True)
data_update.replace("24h","24h-post",inplace=True)
data_update.replace("3h","3h-post",inplace=True)
plot_bar(ax1,data_update,"hilus")
stats_pooled(data_update)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
plt.savefig("hilus-coef.svg")
plt.show()