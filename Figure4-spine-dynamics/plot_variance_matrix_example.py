import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import matplotlib.gridspec as gridspec
import seaborn as sns 
import matplotlib.patches as patches
from scipy import stats

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

def get_unit_matrix_sem(cmatrix,unitsize=10):
    submatrice = []
    for first_spine in np.arange(0,cmatrix.shape[0]-unitsize,1):
        submatrix = cmatrix[first_spine:first_spine+unitsize,first_spine:first_spine+unitsize]
        submatrice.append(submatrix)
    submatrice = np.array(submatrice)
    sem = sp.stats.sem(submatrice,axis=0)
    return sem

def plot_correlation_sem_sample(data):
    group = "control"
    culture = "W3C1"
    segment = "W3C1-segment2"
    segment_data = data[(data["group"]==group) & (data["culture"]==culture) & (data["segment"]==segment)]
    ccs = get_data_matrix_for_each_segment(segment_data)
    sem = get_unit_matrix_sem(ccs)


    fig = plt.figure(figsize=(cm2inch(4), cm2inch(4)))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.1,wspace=0.1)
    ax = plt.subplot(gs[0,0])
    img = ax.pcolor(sem,cmap='binary',vmin=0.,vmax=0.15)

    cax = fig.add_axes([0.92,0.1,0.02,0.8])
    cbar = plt.colorbar(img,cax=cax,orientation='vertical',ticks=[0,0.05,0.1,0.15])
    cbar.set_label(r"$s.e.m$")
    cax.yaxis.set_label_position('right')
    ax.set_xticks([0,5,10,15])
    ax.set_yticks([0,5,10,15])

    plt.savefig("sem_sample.svg")


################ call functions #################

data = pd.read_csv("spine_size_merged.csv")


############# plot cf matrix example ###############
plot_correlation_sem_sample(data)


plt.show()