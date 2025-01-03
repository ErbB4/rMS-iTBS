import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import matplotlib.gridspec as gridspec
import seaborn as sns 
import matplotlib.patches as patches

####figure settings####
plt.rcParams["axes.titlesize"]=10
plt.rcParams["axes.labelsize"]=8
plt.rcParams["axes.linewidth"]=.5
plt.rcParams["lines.linewidth"]=.5
plt.rcParams["lines.markersize"]=1.
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

def get_data_matrix_for_each_segment(segment_data):
    intensities = []
    for spine in np.unique(segment_data["spine_ID"]):
        spine_data = segment_data[segment_data["spine_ID"]==spine]
        intensity = []
        for time in ["day4","day4-h1","day4-h2","day4-h3","day4-h4","day4-h5","day5"]:
            time_data = spine_data[spine_data["time"]==time]
            intensity.append(time_data["RawIntDen"].values)
        intensities.append(intensity)
    intensities = np.array(intensities)
    intensities = intensities.reshape(intensities.shape[0],intensities.shape[1])
    ccs = np.corrcoef(intensities)
    np.fill_diagonal(ccs,np.nan)
    return ccs

def analyze_spine_change_correlation(data,group):
    subdata = data[data["group"]==group]
    averages_meta = []
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]
                ccs = get_data_matrix_for_each_segment(segment_data)
                averaged = average_unit_matrix(ccs)
                averages_meta.append(averaged)
    meta = np.array(averages_meta)
    print(str(group))
    print(meta.shape)
    plt.figure()
    plt.pcolor(np.mean(meta,axis=0),cmap='coolwarm',vmin=-0.15,vmax=0.15)
    plt.colorbar()

def plot_analyze_spine_change_correlation(data,group):
    plt.figure(figsize=(16,9))
    subdata = data[data["group"]==group]
    i=1
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]
                ccs = get_data_matrix_for_each_segment(segment_data)
                averaged = average_unit_matrix(ccs)
                plt.subplot(5,5,i)
                plt.pcolor(ccs,cmap='coolwarm')#,vmin=-0.25,vmax=0.25)
                plt.title(str(group)+str(batch)+str(culture)+str(segment))
                plt.colorbar()

                i = i+1


def plot_increase_only(data,group):
    plt.figure(figsize=(16,9))
    subdata = data[data["group"]==group]
    i=1
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]
                
                ids_to_remove = []
                for spine in np.unique(segment_data["spine_ID"]):
                    if segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day5")]["RawIntDen"].values < segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day4")]["RawIntDen"].values:
                        ids_to_remove.append(spine)
                print(np.max(ids_to_remove))
                print(np.max(segment_data["spine_ID"]))

                ccs = get_data_matrix_for_each_segment(segment_data)
                for spine in ids_to_remove:
                    ccs[:,spine-1] = 0
                    ccs[spine-1,:] = 0
                averaged = average_unit_matrix(ccs)
                plt.subplot(5,5,i)
                plt.pcolor(averaged,cmap='coolwarm',vmin=-0.7,vmax=0.7)
                plt.title(str(group)+str(batch)+str(culture)+str(segment))
                plt.colorbar()

                i = i+1



def plot_two_group_analyze_spine_change_correlation(data,group):
    fig = plt.figure(figsize=(cm2inch(3), cm2inch(3)))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.1,wspace=0.1)
    ax = plt.subplot(gs[0,0])

    subdata = data[data["group"]==group]
    averages_meta = []
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]
                ccs = get_data_matrix_for_each_segment(segment_data)
                averaged = average_unit_matrix(ccs)
                averages_meta.append(averaged)
    meta = np.array(averages_meta)
    img = ax.pcolor(np.mean(meta,axis=0),cmap='coolwarm',vmin=-0.15,vmax=0.15)
    ax.set_title(str(group))
    if group == "control":
        ax.set_xticks([0.5,6.5,14.5])
        ax.set_yticks([0.5,6.5,14.5])
        ax.set_xticklabels(["origin","near", "distant neighbor"])
        ax.set_yticklabels(["origin","near", "distant"],rotation=90)
    if group == "rMS":
        ax.set_xticks([])
        ax.set_yticks([])
        cax = fig.add_axes([0.92,0.1,0.02,0.8])
        cbar = plt.colorbar(img,cax=cax,orientation='vertical')
        cbar.set_label(r"$\bar{cf}$")
        cax.yaxis.set_label_position('right')
    plt.savefig(str(group)+"_matrix_full.svg")


def plot_two_increase(data,group):
    fig = plt.figure(figsize=(cm2inch(3), cm2inch(3)))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.1,wspace=0.1)
    ax = plt.subplot(gs[0,0])

    subdata = data[data["group"]==group]
    averages_meta = []
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]

                ids_to_remove = []
                for spine in np.unique(segment_data["spine_ID"]):
                    if segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day5")]["RawIntDen"].values < segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day4")]["RawIntDen"].values:
                        ids_to_remove.append(spine)
                print(np.max(ids_to_remove))
                print(np.max(segment_data["spine_ID"]))

                ccs = get_data_matrix_for_each_segment(segment_data)
                for spine in ids_to_remove:
                    ccs[:,spine-1] = 0
                    ccs[spine-1,:] = 0

                averaged = average_unit_matrix(ccs)
                averages_meta.append(averaged)
    meta = np.array(averages_meta)
    img = ax.pcolor(np.mean(meta,axis=0),cmap='coolwarm',vmin=-0.15,vmax=0.15)
    ax.set_title(str(group))
    if group == "control":
        ax.set_xticks([])
        ax.set_yticks([])
    if group == "rMS":
        ax.set_xticks([])
        ax.set_yticks([])
        cax = fig.add_axes([0.92,0.1,0.02,0.8])
        cbar = plt.colorbar(img,cax=cax,orientation='vertical')
        cbar.set_label(r"$\bar{cf}$")
        cax.yaxis.set_label_position('right')
    plt.savefig(str(group)+"_matrix_enlarged.svg")


def plot_two_reduced(data,group):
    fig = plt.figure(figsize=(cm2inch(3), cm2inch(3)))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.1,wspace=0.1)
    ax = plt.subplot(gs[0,0])

    subdata = data[data["group"]==group]
    averages_meta = []
    for batch in np.unique(subdata["batch"]):
        batch_data = subdata[subdata["batch"]==batch]
        for culture in np.unique(batch_data["culture"]):
            culture_data = batch_data[batch_data["culture"]==culture]
            for segment in np.unique(culture_data["segment"]):
                segment_data = culture_data[culture_data["segment"]==segment]

                ids_to_remove = []
                for spine in np.unique(segment_data["spine_ID"]):
                    if segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day5")]["RawIntDen"].values > segment_data.loc[(segment_data["spine_ID"]==spine) & (segment_data["time"]=="day4")]["RawIntDen"].values:
                        ids_to_remove.append(spine)
                print(np.max(ids_to_remove))
                print(np.max(segment_data["spine_ID"]))

                ccs = get_data_matrix_for_each_segment(segment_data)
                for spine in ids_to_remove:
                    ccs[:,spine-1] = 0
                    ccs[spine-1,:] = 0

                averaged = average_unit_matrix(ccs)
                averages_meta.append(averaged)
    meta = np.array(averages_meta)
    img = ax.pcolor(np.mean(meta,axis=0),cmap='coolwarm',vmin=-0.15,vmax=0.15)
    ax.set_title(str(group))
    if group == "control":
        ax.set_xticks([])
        ax.set_yticks([])
    if group == "rMS":
        ax.set_xticks([])
        ax.set_yticks([])
        cax = fig.add_axes([0.92,0.1,0.02,0.8])
        cbar = plt.colorbar(img,cax=cax,orientation='vertical')
        cbar.set_label(r"$\bar{cf}$")
        cax.yaxis.set_label_position('right')
    plt.savefig(str(group)+"_matrix_reduced.svg")


def average_unit_matrix(cmatrix,unitsize=15):
    submatrice = []
    for first_spine in np.arange(0,cmatrix.shape[0]-unitsize,1):
        submatrix = cmatrix[first_spine:first_spine+unitsize,first_spine:first_spine+unitsize]
        submatrice.append(submatrix)
    submatrice = np.array(submatrice)
    averaged = np.mean(submatrice,axis=0)
    return averaged
    #plt.pcolor(averaged,cmap='coolwarm')



def plot_spine_size_trace(data):
    group = "control"
    culture = "W3C1"
    segment = "W3C1-segment2"
    trace_set = data[(data["group"]==group) & (data["culture"]==culture) & (data["segment"]==segment)  & (data["time_cor"]>3.)]

    fig = plt.figure(figsize=(cm2inch(5), cm2inch(4.3)))
    gs = gridspec.GridSpec(5,1)
    gs.update(top=0.98,bottom=0.01,left=0.15,right=0.98,hspace=0.1,wspace=0.0)
    
    ax_objs = []
    i=0
    for spine in [1,2,10,15,30]:
        trace = trace_set[(trace_set["spine_ID"]==spine)].sort_values(by="time_cor")
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        plot = (ax_objs[-1].plot(trace.time_cor,trace.normed_size,'.-',color='k',linewidth=.5))

        if i == 4:
            ax_objs[-1].set_yticks([])
            #pass
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].axis("off")

        ax_objs[-1].spines['right'].set_visible(False)
        ax_objs[-1].spines['top'].set_visible(False)
        ax_objs[-1].yaxis.set_ticks_position('left')
        ax_objs[-1].xaxis.set_ticks_position('bottom')
        ax_objs[-1].set_ylim()
        i = i+1

    plt.savefig("spine_trace_sample.svg")


def plot_spine_size_trace_raw(data):
    group = "control"
    culture = "W3C1"
    segment = "W3C1-segment2"
    trace_set = data[(data["group"]==group) & (data["culture"]==culture) & (data["segment"]==segment)  & (data["time_cor"]>3.)]

    fig = plt.figure(figsize=(cm2inch(5.5), cm2inch(2.5)))
    gs = gridspec.GridSpec(1,1)
    gs.update(top=0.98,bottom=0.1,left=0.15,right=0.98,hspace=0.1,wspace=0.0)
    ax1 = plt.subplot(gs[0,0])
    i=0
    shades=np.linspace(0.1,1,5)
    for spine in [1,2,10,15]:
        trace = trace_set[(trace_set["spine_ID"]==spine)].sort_values(by="time_cor")
        plot = (ax1.plot(trace.time_cor.values,trace.RawIntDen.values,'.-',color='k',linewidth=.5,alpha=shades[i],label="s "+str(spine)))
        i = i+1
    ax1.set_xticklabels([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(np.unique(trace_set["time_cor"]))
    ax1.set_xticklabels(["day 4", "1h", "2h", "3h", "4h", "5h", "24h"],rotation=90)
    ax1.set_ylabel("spine size (a.u.)")
    ax1.axvline(x=4.1,ymin=0,ymax=0.75,linestyle='--',color='#ea638c',linewidth=0.5)
    ax1.text(4.1,16000,"iTBS-rMS\nif applies",fontsize=5,color='#ea638c',ha="left",va="bottom")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.legend(loc='best',frameon=False)

    plt.savefig("spine_trace_sample.svg")


def plot_correlation_sample(data):
    group = "control"
    culture = "W3C1"
    segment = "W3C1-segment2"
    segment_data = data[(data["group"]==group) & (data["culture"]==culture) & (data["segment"]==segment)]
    ccs = get_data_matrix_for_each_segment(segment_data)

    fig = plt.figure(figsize=(cm2inch(4), cm2inch(4)))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.1,wspace=0.1)
    ax = plt.subplot(gs[0,0])
    img = ax.pcolor(ccs,cmap='coolwarm',vmin=-1.,vmax=1.)

    ax.set_xticks([0.5,9.5,14.5])
    ax.set_yticks([0.5,9.5,14.5])
    ax.set_xticklabels(["s1", "s10","s15"])
    ax.set_yticklabels(["s1", "s10","s15"])
    cax = fig.add_axes([0.92,0.1,0.02,0.8])
    cbar = plt.colorbar(img,cax=cax,orientation='vertical')
    cbar.set_label(r"$cf$")
    cax.yaxis.set_label_position('right')

    rect1 = patches.Rectangle((1, 1), 15, 15, linewidth=.5, edgecolor='k', facecolor='none')
    rect2 = patches.Rectangle((11, 11), 15, 15, linewidth=.5, edgecolor='k', facecolor='none')
    rect3 = patches.Rectangle((12, 12), 15, 15, linewidth=.5, edgecolor='k', facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    ax.text(28,28,"unit matrix",fontsize=6,ha='center',va="bottom",color="k")

    plt.savefig("CF_sample.svg")





def average_unit_matrix(cmatrix,unitsize=15):
    submatrice = []
    for first_spine in np.arange(0,cmatrix.shape[0]-unitsize,1):
        submatrix = cmatrix[first_spine:first_spine+unitsize,first_spine:first_spine+unitsize]
        submatrice.append(submatrix)
    submatrice = np.array(submatrice)
    averaged = np.mean(submatrice,axis=0)
    return averaged
    #plt.pcolor(averaged,cmap='coolwarm')


################ call functions #################

data = pd.read_csv("spine_size_merged.csv")

############# plot spine size traces for s1,s2,s10,s15 ##############
plot_spine_size_trace_raw(data)


############# plot cf matrix example ###############
plot_correlation_sample(data)


############# plot averaged matrix examples ###############
plot_two_group_analyze_spine_change_correlation(data,"control")
plot_two_group_analyze_spine_change_correlation(data,"rMS")


############# plot averaged matrix examples for increased size only ###############
plot_two_increase(data,"control")
plot_two_increase(data,"rMS")

############# plot averaged matrix examples for reduced size only ###############
plot_two_reduced(data,"control")
plot_two_reduced(data,"rMS")


plt.show()