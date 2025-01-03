import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#loading file
import pickle
res_name = 'DG_itbs_data_res'
with open(res_name, 'rb') as f:
    results = pickle.load(f)


def rescale_and_detrend_trace(intensity):
    data = pd.DataFrame({"RawIntDen": intensity})
    rolling_window = 20
    data["RawIntDen_median"] = data["RawIntDen"].rolling(rolling_window).median()
    data["RawIntDen_trend"] = data["RawIntDen_median"].fillna(
        data.RawIntDen_median.iloc[rolling_window-1]
    )
    data["RawIntDen_detrend"] = data.RawIntDen - data.RawIntDen_trend
    data["RawIntDen_rescaled"] = data.RawIntDen_detrend/np.mean(data.RawIntDen_trend)
    return data["RawIntDen_rescaled"].values, data["RawIntDen_trend"].values

def plot_example(prep_date,group,culture,neuron_id,color,ax):
    cdata = results[prep_date][group][culture]["time_series"][neuron_id]
    x = np.linspace(0,120,len(cdata))
    data_new = rescale_and_detrend_trace(cdata)[0]
    ax.plot(x,data_new,linestyle='-',color=color)
    ax.axhline(y=np.mean(data_new)+3*np.std(data_new),linestyle='--',color='#2d6a4f')

def plot_process(prep_date,group,culture,neuron_id,ax1,ax2,ax3):
    cdata = results[prep_date][group][culture]["time_series"][neuron_id]
    x = np.linspace(0,120,len(cdata))
    data_new,trend = rescale_and_detrend_trace(cdata)
    ax1.plot(x,data_new,linestyle='-',color='#293241',label='processed')
    ax1.axhline(y=np.mean(data_new)+3*np.std(data_new),linestyle='--',color='#2d6a4f')
    ax2.plot(x,data_new,linestyle='-',color='#293241',label='processed')
    ax2.axhline(y=0,linestyle='--',color='grey',label='average trend')
    ax2.fill_between(x,y1=0,y2=data_new,color='#ffaaaaff')
    ax3.plot(x,data_new,linestyle='-',color='#293241',label='processed')
    ax3.axhline(y=0,linestyle='--',color='grey',label='average trend')
    ax3.fill_between(x,y1=0,y2=data_new,color='#ffaaaaff')
    ax3.set_xlim(0,20)

    ax2.plot([0,30],[np.min(data_new),np.min(data_new)],'-',color='k',linewidth=1.)


plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["lines.linewidth"]=0.5
plt.rcParams["lines.markersize"]=5
plt.rcParams["xtick.labelsize"]=6
plt.rcParams["ytick.labelsize"]=6
plt.rcParams["legend.fontsize"]=6

plt.rcParams.update({'font.family':'arial'})
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def cm2inch(value):
    return value/2.54


fig = plt.figure(figsize=(cm2inch(7.5), cm2inch(5)))
gs1 = gridspec.GridSpec(6, 1)
gs1.update(top=0.99,bottom=0.0,left=0.0,right=1.,hspace=0.0,wspace=0.0)
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[2,0])
ax4 = plt.subplot(gs1[3,0])
ax5 = plt.subplot(gs1[4,0])
ax6 = plt.subplot(gs1[5,0])

ax3 = fig.add_axes([0.2,0.67,0.6,0.2])
ax3.set_xticks([])
ax3.set_yticks([])

plot_process("07-04-2022","24h","video-4-1-1.czi",1,ax1,ax2,ax3)
plot_example("07-04-2022","3h","Video-4-1-2.czi",10,'#293241ff',ax4) #3h control
plot_example("07-04-2022","3h","Video-4-2-1.czi",4,'#98c1d9ff',ax5) #3h rMS
plot_example("13-04-2022","24h-post","Video-6-1-2.czi",65,'#ee6c4dff',ax6) #24h rMS



for ax in [ax1,ax2,ax4,ax5,ax6]:
    ax.axis("off")
#ax1.set_ylim(-500,3200)
plt.savefig("method-DG-trace.svg")
plt.show()

