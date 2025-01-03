import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.gridspec as gridspec
import pingouin as pg 
from scipy import stats
import seaborn as sns 
from scipy.stats import pearsonr


def reorganize_data(data):
    pre  = data[data["time"]=="day4"]
    post = data[data["time"]=="day5"]

    new_data = pre.loc[:,["spine_ID","RawIntDen","batch","group","segment","culture"]]
    new_data["post_RawIntDen"] = post["RawIntDen"].values
    new_data.rename(columns={'RawIntDen':'pre_RawIntDen'}, inplace=True)
    return new_data


def get_spine_density_change(datadensity,segment_ID):
    subdata = datadensity[datadensity["segment"]==segment_ID]
    diffs   = subdata[subdata["time"]=="24h"]["spine_density"].values - subdata[subdata["time"]=="day4"]["spine_density"].values
    return diffs

def get_average_spine_size_change(datasize,segment_ID):
    subdata = datasize[datasize["segment"]==segment_ID]
    diffs   = subdata["post_RawIntDen"].values - subdata["pre_RawIntDen"].values
    return np.mean(diffs),sp.stats.sem(diffs)


def annotate_density_in_target_files(datasize,datadensity):
    for name in np.unique(datadensity["segment"]):
        diffs_size = get_average_spine_size_change(datasize,name)
        diffs_density = get_spine_density_change(datadensity,name)
        datadensity.loc[datadensity["segment"]==name,"spine_size_diff"]     = diffs_size[0]
        datadensity.loc[datadensity["segment"]==name,"spine_size_diff_sem"] = diffs_size[1]
        datadensity.loc[datadensity["segment"]==name,"spine_density_diff"]  = diffs_density[0]
        datadensity.loc[datadensity["segment"]==name,"spine_density_diff_abs"]  = abs(diffs_density[0])
        datadensity.replace('sham','control', inplace=True)

    return datadensity


def plot_fit_curves(data,groupname,ax,palette):
    theta = np.polyfit(data["spine_density"].values, data["spine_size_diff"].values, 1)
    cov = pearsonr(data["spine_density"].values,data["spine_size_diff"].values)
    print(cov)
    if cov[1]>0.05:
        pass
    else: 
        x = np.arange(0.5,3.0,0.1)
        if groupname=="control":
            color = palette[0]
        if groupname=="rMS":
            color = palette[1]

        ax.plot(x,theta[1]+theta[0]*x,'-',color=color)

####figure settings####
plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["axes.linewidth"]=.5
plt.rcParams["lines.linewidth"]=1.5
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
size_raw = data_raw.copy()
size = reorganize_data(size_raw)

data_raw = pd.read_csv("./spine_density_old.csv")
density = data_raw.copy()
data_all = annotate_density_in_target_files(size,density)
data_half = data_all[data_all["time"]=="day4"]

fig = plt.figure(figsize=(cm2inch(4.5), cm2inch(4.5)))
gs = gridspec.GridSpec(1, 1)
gs.update(top=0.9,bottom=0.23,left=0.23,right=0.9,hspace=0.1,wspace=0.15)
ax = plt.subplot(gs[0,0])

palette = ['#293241','#98c1d9']
sns.scatterplot(data=data_half,x="spine_density",y="spine_size_diff",hue="group",hue_order=["control","rMS"],palette=palette,ax=ax)

#fit curves
plot_fit_curves(data_half,"control",ax,palette)
plot_fit_curves(data_half,"rMS",ax,palette)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xlabel(r"Baseline spine density (per $\mu$m)")
ax.set_ylabel(r"$\bar{\Delta}$ spine size (a.u.)")
ax.axhline(y=0,linestyle='--',color='grey',linewidth=0.5)
#ax.text(1.8,5400,r'$r(24)=0.65$',fontsize=8)

ax.set_xlim(0.5,3.0)
ax.set_ylim(-6500,6500)
plt.savefig("size_and_density_all.svg")
plt.show()