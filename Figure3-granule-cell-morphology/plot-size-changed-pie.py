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


def plot_size_change(data,groupname,ax1,ax2,colors1,colors2):
    groupdata = data[data["group"]==groupname]
    day4  = groupdata[groupdata["time"]=="day4"]
    day4_h2 = groupdata[groupdata["time"]=="day4-h2"]
    day5 = groupdata[groupdata["time"]=="day5"]

    diffs_1 = day4_h2["RawIntDen"].values - day4["RawIntDen"].values
    diffs_2 = day5["RawIntDen"].values - day4_h2["RawIntDen"].values

    new_data = day4.loc[:,["batch","culture","group","time","spine_ID","RawIntDen"]]
    new_data["changes_at_2h_from_day4"] = diffs_1
    new_data["changes_at_day5_from_2h"] = diffs_2

    # potentiated at 2h:
    po_num = len(new_data[new_data["changes_at_2h_from_day4"]>0])
    de_num = len(new_data[new_data["changes_at_2h_from_day4"]<0])

    # potentiated at day5 from 2h potentiated ones
    po_po_num = len(new_data.loc[(new_data["changes_at_2h_from_day4"]>0) & (new_data["changes_at_day5_from_2h"]>0)])
    po_ne_num = len(new_data.loc[(new_data["changes_at_2h_from_day4"]>0) & (new_data["changes_at_day5_from_2h"]<0)])

    # potentiated at day5 from 2h depressed ones
    ne_po_num = len(new_data.loc[(new_data["changes_at_2h_from_day4"]<0) & (new_data["changes_at_day5_from_2h"]>0)])
    ne_ne_num = len(new_data.loc[(new_data["changes_at_2h_from_day4"]<0) & (new_data["changes_at_day5_from_2h"]<0)])

    ax1.pie([po_num, de_num],colors=colors1,explode=[0.2,0.],labels=["enlarged","shrunk"])
    ax2.pie([po_po_num,po_ne_num,ne_po_num,ne_ne_num],colors=colors2,explode=[0.2,0.2,0.,0.],labels=["2x enlagred","shrunk","enlarged","2x shrunk"])

    ax1.set_title("Day 4 to 2h-post",fontsize=8)
    ax2.set_title("2h- to 24h-post",fontsize=8)

    print(po_num, de_num, po_po_num,po_ne_num,ne_po_num,ne_ne_num)

    return po_num, de_num, po_po_num,po_ne_num,ne_po_num,ne_ne_num



def plot_pie_chart(data):
    fig = plt.figure(figsize=(cm2inch(6), cm2inch(3)))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.9,bottom=0.1,left=0.05,right=0.85,hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs1[0,0])
    ax2 = plt.subplot(gs1[0,1])

    fig = plt.figure(figsize=(cm2inch(6), cm2inch(3)))
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.9,bottom=0.1,left=0.05,right=0.85,hspace=0.1,wspace=0.1)
    ax3 = plt.subplot(gs2[0,0])
    ax4 = plt.subplot(gs2[0,1])

    plot_size_change(data,"control",ax1,ax2,colors1=["#6c757d","#aaacad"],colors2=["#6c757d","#aaacad","#6c757d","#aaacad"])
    plot_size_change(data,"rMS",ax3,ax4,colors1=["#61A1D7","#a3ccf0"],colors2=["#61A1D7","#a3ccf0","#61A1D7","#a3ccf0"])


plot_pie_chart(data)
plt.figure(1)
plt.savefig("control_piechart.svg")
plt.figure(2)
plt.savefig("rMS_piechart.svg")

plt.show()