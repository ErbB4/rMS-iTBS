import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.gridspec as gridspec
import seaborn as sns 
from scipy import stats
import pickle
from scipy import signal
from numpy import trapz

####figure settings####
plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["axes.linewidth"]=.5
plt.rcParams["lines.linewidth"]=0.5
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
        time_1 = culture_data.loc[culture_data["timing"]=="pre"].dom_power
        time_2 = culture_data.loc[culture_data["timing"]=="3h-post"].dom_power
        time_3 = culture_data.loc[culture_data["timing"]=="24h-post"].dom_power

        means[0,idx] = np.mean(time_1)
        means[1,idx] = np.mean(time_2)
        means[2,idx] = np.mean(time_3)
    return means

def get_means_individual_two_time_points(subdata):
#################### get means for plot ##############
    means = np.zeros([2,len(np.unique(subdata.culture))])
    for idx,culture in enumerate(np.unique(subdata["culture"])):
        culture_data = subdata[subdata["culture"]==culture]
        time_2 = culture_data.loc[culture_data["timing"]=="3h-post"].dom_power
        time_3 = culture_data.loc[culture_data["timing"]=="24h-post"].dom_power

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
    if len(np.unique(control_data.timing))==3:
        means_control = np.array([np.mean(control_data[control_data["timing"]=="pre"].dom_power),np.mean(control_data[control_data["timing"]=="3h-post"].dom_power),np.mean(control_data[control_data["timing"]=="24h-post"].dom_power)])
        means_rMS = np.array([np.mean(rMS_data[rMS_data["timing"]=="pre"].dom_power),np.mean(rMS_data[rMS_data["timing"]=="3h-post"].dom_power),np.mean(rMS_data[rMS_data["timing"]=="24h-post"].dom_power)])
    if len(np.unique(control_data.timing))==2:
        means_control = np.array([np.mean(control_data[control_data["timing"]=="3h-post"].dom_power),np.mean(control_data[control_data["timing"]=="24h-post"].dom_power)])
        means_rMS = np.array([np.mean(rMS_data[rMS_data["timing"]=="3h-post"].dom_power),np.mean(rMS_data[rMS_data["timing"]=="24h-post"].dom_power)])
    return means_control,means_rMS

def stats_pooled(batchdata):
#################### stats ##############
    control_data = batchdata[batchdata["group"]=="control"]
    rMS_data = batchdata[batchdata["group"]=="rMS"]
    print("3h-stats")
    print(sp.stats.mannwhitneyu(control_data[control_data["timing"]=="3h-post"].dom_power,rMS_data[rMS_data["timing"]=="3h-post"].dom_power))
    print("24h-stats")
    print(sp.stats.mannwhitneyu(control_data[control_data["timing"]=="24h-post"].dom_power,rMS_data[rMS_data["timing"]=="24h-post"].dom_power))


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



#################### load data and plot ##############
data = pd.read_csv("calcium_data_hilus.csv")
data =data.dropna()

################### plot example power spectrum and averaged #########
fig = plt.figure(1,figsize=(cm2inch(4.3), cm2inch(3.7)))
gs1 = gridspec.GridSpec(3, 1)
gs1.update(top=0.99,bottom=0.2,left=0.1,right=0.98,hspace=0.1,wspace=0.1)
ax1 = plt.subplot(gs1[1:3,0])
ax2 = plt.subplot(gs1[0:1,0])


def get_power_spectrum(series):
    fs = []
    Ss = []
    for i in series.keys():
        single_series = series[i]
        f,S = sp.signal.periodogram(single_series, 1./(120./len(single_series)), scaling = 'density')
        fs.append(f)
        Ss.append(S)
    fs = np.array(fs)
    Ss = np.array(Ss)
    return fs,Ss, np.mean(Ss,axis=0)

res_name = 'hilus_itbs_data_res'
with open(res_name, 'rb') as f:
    results = pickle.load(f)

series = results["07-04-2022"]["24h-from-3h"]["video-2-2-1.czi"]["time_series"]
matrix = np.array([series[i] for i in series.keys()])
fs,Ss, ave = get_power_spectrum(series)
ax1.imshow(Ss,vmax=1000,vmin=0.,aspect=1.)
ax1.set_xlabel("Freq. (Hz)")
ax1.set_ylabel("neuron ID")
ax1.set_xticks([0,120,240,360])
ax1.set_xticklabels([0,1,2,3])

ax2.plot(fs[0],ave,'k-',linewidth=.5)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_ylabel("ave.\npower")
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_xlim(fs[0][0],fs[0][-1])

############ averaged per group ###########
fig = plt.figure(2,figsize=(cm2inch(4), cm2inch(3.7)))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.15,wspace=0.15)
ax3 = plt.subplot(gs1[0,0])

sham_3h_traces=[]
sham_24h_traces=[]

rMS_3h_traces=[]
rMS_24h_traces=[]

for culture in results["07-04-2022"]["24h-from-3h"].keys():
    if culture[6:-4] in ["1-2-1","1-2-2","2-2-1","2-2-2","3-2-1","3-2-2"]:
        series = results["07-04-2022"]["24h-from-3h"][culture]["time_series"]
        rMS_24h_traces.append(get_power_spectrum(series)[2])
    if culture[6:-4] in ["1-1-1","2-1-1","2-1-2","3-1-1","3-1-2"]:
        series = results["07-04-2022"]["24h-from-3h"][culture]["time_series"]
        sham_24h_traces.append(get_power_spectrum(series)[2])

for culture in results["07-04-2022"]["3h"].keys():
    if culture[6:-4] in ["1-2-1","1-2-2","2-2-1","2-2-2","3-2-1","3-2-2"]:
        series = results["07-04-2022"]["3h"][culture]["time_series"]
        rMS_3h_traces.append(get_power_spectrum(series)[2])
    if culture[6:-4] in ["1-1-1","2-1-1","2-1-2","3-1-1","3-1-2"]:
        series = results["07-04-2022"]["3h"][culture]["time_series"]
        sham_3h_traces.append(get_power_spectrum(series)[2])

for culture in results["13-04-2022"]["24h-post"].keys():
    if culture[6:-4] in ["4-1-2","5-1-2","6-1-2"]:
        series = results["13-04-2022"]["24h-post"][culture]["time_series"]
        rMS_24h_traces.append(get_power_spectrum(series)[2])
    if culture[6:-4] in ["4-2-1","4-2-2","5-2-1","5-2-2","6-2-1","6-2-2"]:
        series = results["13-04-2022"]["24h-post"][culture]["time_series"]
        sham_24h_traces.append(get_power_spectrum(series)[2])

for culture in results["13-04-2022"]["3h-post"].keys():
    if culture[6:-4] in ["4-1-2","5-1-2","6-1-2"]:
        series = results["13-04-2022"]["3h-post"][culture]["time_series"]
        rMS_3h_traces.append(get_power_spectrum(series)[2])
    if culture[6:-4] in ["4-2-1","4-2-2","5-2-1","5-2-2","6-2-1","6-2-2"]:
        series = results["13-04-2022"]["3h-post"][culture]["time_series"]
        sham_3h_traces.append(get_power_spectrum(series)[2])

sham_3h_traces = np.array(sham_3h_traces)
sham_24h_traces = np.array(sham_24h_traces)
rMS_3h_traces = np.array(rMS_3h_traces)
rMS_24h_traces = np.array(rMS_24h_traces)

offset = 100
ax3.plot(fs[0],rMS_3h_traces.mean(axis=0)+offset,'-',color='#98c1d9',label="3h-post")
ax3.fill_between(fs[0],rMS_3h_traces.mean(axis=0)+sp.stats.sem(rMS_3h_traces,axis=0)+offset,rMS_3h_traces.mean(axis=0)-sp.stats.sem(rMS_3h_traces,axis=0)+offset,color='#98c1d9',alpha=0.2)

ax3.plot(fs[0],sham_3h_traces.mean(axis=0)+offset,'-',color='#293241',linewidth=0.5,label="control")
ax3.fill_between(fs[0],sham_3h_traces.mean(axis=0)+sp.stats.sem(sham_3h_traces,axis=0)+offset,sham_3h_traces.mean(axis=0)-sp.stats.sem(sham_3h_traces,axis=0)+offset,color='#293241',alpha=0.2)

ax3.plot(fs[0],-1*rMS_24h_traces.mean(axis=0),'-',color='#ee6c4d',label="24h-post")
ax3.fill_between(fs[0],-1*(rMS_24h_traces.mean(axis=0)+sp.stats.sem(rMS_24h_traces,axis=0)),-1*(rMS_24h_traces.mean(axis=0)-sp.stats.sem(rMS_24h_traces,axis=0)),color='#ee6c4d',alpha=0.2)

ax3.plot(fs[0],-1*sham_24h_traces.mean(axis=0),'-',color='#293241',linewidth=0.5,label="control")
ax3.fill_between(fs[0],-1*(sham_24h_traces.mean(axis=0)+sp.stats.sem(sham_24h_traces,axis=0)),-1*(sham_24h_traces.mean(axis=0)-sp.stats.sem(sham_24h_traces,axis=0)),color='#293241',alpha=0.2)
ax3.legend(loc="best",frameon=False)
for ax in [ax3]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax3.set_ylabel("ave. power (a.u.)")
    ax.set_xlabel("Freq. (Hz)")
    ax.set_xlim(-0.03,0.5)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5])

plt.figure(1)
plt.savefig("hilus_heatmap.svg")
plt.figure(2)
plt.savefig("hilus_averaged_trace_power.svg")
plt.show()


def stats_AUC(data1,data2,data3,data4):
    x = np.linspace(0,120,data1.shape[1])
    AUCs_1 = []
    AUCs_2 = []
    AUCs_3 = []
    AUCs_4 = []

    for trace in data1:
        AUCs_1.append(trapz(y=trace,x=x))
    for trace in data2:
        AUCs_2.append(trapz(y=trace,x=x))
    for trace in data3:
        AUCs_3.append(trapz(y=trace,x=x))
    for trace in data4:
        AUCs_4.append(trapz(y=trace,x=x))
    print(sp.stats.mannwhitneyu(AUCs_1,AUCs_2))
    print(sp.stats.mannwhitneyu(AUCs_3,AUCs_4))
    return AUCs_1,AUCs_2,AUCs_3,AUCs_4

AUCs_1,AUCs_2,AUCs_3,AUCs_4 = stats_AUC(sham_3h_traces,rMS_3h_traces,sham_24h_traces,rMS_24h_traces)

plt.show()


import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("calcium_data_hilus.csv")
data =data.dropna()

data_update = data[data["prep_date"]!="15-03-2022"]
data_update.replace("24h-from-3h","24h-post",inplace=True)
data_update.replace("24h","24h-post",inplace=True)
data_update.replace("3h","3h-post",inplace=True)


data_3h = data_update[data_update["timing"]=="3h-post"]
data_24h = data_update[data_update["timing"]=="24h-post"]

md1 = smf.mixedlm("power_auc ~ group", data_3h, groups=data_3h["culture"])
mdf1 = md1.fit()

md2 = smf.mixedlm("power_auc ~ group", data_24h, groups=data_24h["culture"])
mdf2 = md2.fit()

print(mdf1.summary()) #not significant
print(mdf2.summary()) #not significant