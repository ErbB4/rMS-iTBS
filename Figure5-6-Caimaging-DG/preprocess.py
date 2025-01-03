#loading file
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp  
from scipy import signal


# different combinations of preprocessing was defined here

def detrend_trace(intensity):
	#only detrend the raw trace
    data = pd.DataFrame({"RawIntDen": intensity})
    rolling_window = 20
    data["RawIntDen_median"] = data["RawIntDen"].rolling(rolling_window).median()
    data["RawIntDen_trend"] = data["RawIntDen_median"].fillna(
        data.RawIntDen_median.iloc[rolling_window-1]
    )
    data["RawIntDen_detrend"] = data.RawIntDen - data.RawIntDen_trend
    return data["RawIntDen_detrend"].values


def rescale_and_detrend_trace(intensity):
	#detrend and rescale the trace
    data = pd.DataFrame({"RawIntDen": intensity})
    rolling_window = 20
    data["RawIntDen_median"] = data["RawIntDen"].rolling(rolling_window).median()
    data["RawIntDen_trend"] = data["RawIntDen_median"].fillna(
        data.RawIntDen_median.iloc[rolling_window-1]
    )
    data["RawIntDen_detrend"] = data.RawIntDen - data.RawIntDen_trend
    data["RawIntDen_rescaled"] = data.RawIntDen_detrend/np.mean(data.RawIntDen_trend)
    return data["RawIntDen_rescaled"].values


def moving_smoothing(intensity_series):
	#moving average the trace with a small window size 2
    data = pd.DataFrame({"Intensity": intensity_series})
    rolling_window = 2
    data["Intensity_mean"] = data["Intensity"].rolling(rolling_window).mean()
    data["Intensity_smoothed"] = data["Intensity_mean"].fillna(
        data.Intensity_mean.iloc[rolling_window-1]
    )
    return data["Intensity_smoothed"].values


def get_trace_stats(intensity_series,threshold_ratio=3):
	#this function was defined to detect spikes for trace after different proprocessing
    data = pd.DataFrame({"Intensity": intensity_series})
    data["reference"] = (
        data.Intensity.mean() + threshold_ratio * data.Intensity.std()
    )
    data["is_above_threshold"] = (data.Intensity >= data.reference).astype(int)
    data = data.reset_index().rename(columns={"index": "time_order"})
    data["is_above_threshold_change"] = data.is_above_threshold.diff()
    spike_boundary_index = data.loc[
        data.is_above_threshold_change != 0
    ].dropna().time_order
    data["time"] = data.time_order/len(data)*120
    firing_rate = int(len(spike_boundary_index) / 2) / 120
    return int(len(spike_boundary_index) / 2), firing_rate #spike numbers and FR

def analyze_trace(raw_intensity,type,threshold_ratio=3):
	#get the spike numbers or FR under different preprocessing method
    if type == "raw":
        spike_n = get_trace_stats(raw_intensity,threshold_ratio=3)[0]
    if type == "detrend":
        detrended_intensity = detrend_trace(raw_intensity)
        spike_n = get_trace_stats(detrended_intensity,threshold_ratio=3)[0]
    if type == "detrend_smooth":
        detrended_intensity = detrend_trace(raw_intensity)
        smoothed_intensity = moving_smoothing(detrended_intensity)
        spike_n = get_trace_stats(smoothed_intensity,threshold_ratio=3)[0]
    if type == "rescale_detrend":
        rescaled_intensity = rescale_and_detrend_trace(raw_intensity)
        spike_n = get_trace_stats(rescaled_intensity,threshold_ratio=3)[0]
    if type == "rescale_detrend_smooth":
        rescaled_intensity = rescale_and_detrend_trace(raw_intensity)
        smoothed_intensity = moving_smoothing(rescaled_intensity)
        spike_n = get_trace_stats(smoothed_intensity,threshold_ratio=3)[0]
    return spike_n


def plot_and_compare():
    #quality control for the trace and preprocessing methods
    for prep_date in results.keys():
        for group in results[prep_date].keys():
            for culture in results[prep_date][group].keys():
                for neuron in results[prep_date][group][culture]["time_series"].keys():
                    raw_intensity = results[prep_date][group][culture]["time_series"][neuron]

                    x = np.linspace(0,120,len(raw_intensity))
                    m1 = detrend_trace(raw_intensity)
                    m2 = moving_smoothing(m1)
                    m3 = rescale_and_detrend_trace(raw_intensity)
                    m4 = moving_smoothing(m3)

                    plt.figure()
                    plt.subplot(311)
                    plt.plot(x,raw_intensity,color='k',label='raw')
                    n = analyze_trace(raw_intensity,"raw")
                    plt.title(str(n) + " spikes",fontsize=25)
                    plt.xticks([])

                    plt.subplot(312)
                    plt.plot(x,m3,color='r',label='rescale_detrend')
                    n = analyze_trace(raw_intensity,"rescale_detrend")
                    plt.title(str(n) + " spikes",fontsize=25)
                    plt.xticks([])

                    plt.subplot(313)
                    plt.plot(x,m4,color='blue',label='rescaled and smoothed')
                    n = analyze_trace(raw_intensity,"rescale_detrend_smooth")
                    plt.title(str(n) + " spikes",fontsize=25)
                    plt.xticks([])
                    
                    plt.savefig("./raw_traces/hilus/"+str(prep_date)+"/"+str(group)+"/"+str(culture)[6:-4]+"/"+str(neuron)+".png")
                    plt.figure().clear()


def plot_power_spectrum(results):
    #quality control for the trace and preprocessing methods
    for prep_date in results.keys():
        for group in results[prep_date].keys():
            for culture in results[prep_date][group].keys():
                series = results[prep_date][group][culture]["time_series"]
                if len(series)<2:
                    pass
                else:
                    fs,Ss = get_power_spectrum(series)
                    plt.figure()
                    plt.plot(fs[0],np.mean(Ss,axis=0))
                    plt.xlabel("Freq. (Hz)")
                    plt.ylabel("neuron ID")
                    f = fs[0]
                    ave = np.mean(Ss,axis=0)
                    print("dominant frequency")
                    print(f[np.where(ave==np.max(ave))])
                    print(np.max(ave))
                    print("=================")

                    plt.savefig("./raw_traces/DG/"+str(prep_date)+"/"+str(group)+"/"+str(culture)[6:-4]+"/"+"PS_sum.png")
                    plt.figure().clear()


def get_meta_raw_trace_plot(results):
    for prep_date in results.keys():
        for group in results[prep_date].keys():
            for culture in results[prep_date][group].keys():
                series = results[prep_date][group][culture]["time_series"]
                if len(series)==1:
                    pass
                else:
                    matrix = np.array([series[i] for i in series.keys()])
                    meta_time_course = np.mean(matrix,axis=0)
                    re_de = rescale_and_detrend_trace(meta_time_course)

                    plt.figure()
                    x = np.linspace(0,120.,matrix.shape[1])
                    plt.subplot(211)
                    plt.plot(x,meta_time_course,color='k',label='meta')
                    plt.subplot(212)
                    plt.plot(x,re_de,color='r',label='meta_detrended')
                    n = analyze_trace(meta_time_course,"rescale_detrend")
                    plt.title(str(n) + " spikes",fontsize=25)
                    plt.xticks([])
                    plt.savefig("./raw_traces/DG/"+str(prep_date)+"/"+str(group)+"/"+str(culture)[6:-4]+"/"+"meta_trace.png")
                    plt.figure().clear()    

def get_meta_raw_trace_sub(results):
    for prep_date in results.keys():
        for group in results[prep_date].keys():
            for culture in results[prep_date][group].keys():
                series = results[prep_date][group][culture]["time_series"]
                if len(series)==1:
                    pass
                else:
                    matrix = np.array([rescale_and_detrend_trace(series[i]) for i in series.keys()])
                    meta_time_course = np.mean(matrix,axis=0)
                    re_de = rescale_and_detrend_trace(meta_time_course)

                    sample_neuron = rescale_and_detrend_trace(matrix[1])

                    plt.figure()
                    x = np.linspace(0,120.,matrix.shape[1])
                    plt.subplot(411)
                    plt.plot(x,meta_time_course,color='k',label='meta')
                    plt.subplot(412)
                    plt.plot(x,re_de,color='r',label='meta_detrended')
                    n = analyze_trace(re_de,"raw")
                    plt.title(str(n) + " spikes",fontsize=25)
                    plt.xticks([])
                    plt.subplot(413)
                    plt.plot(x,sample_neuron,color='grey')
                    plt.subplot(414)
                    plt.plot(x,sample_neuron-re_de,color='grey')
                    print(analyze_trace(sample_neuron-re_de,"raw"))
                    print(analyze_trace(sample_neuron,"raw"))

                    plt.savefig("./raw_traces/DG/"+str(prep_date)+"/"+str(group)+"/"+str(culture)[6:-4]+"/"+"meta_trace.png")
                    plt.figure().clear() 


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
    return fs,Ss


def add_group_info(data):
#################### add group info ##############
    data.loc[(data["prep_date"]=="13-04-2022") & (data["culture"].isin(["4-1-2","5-1-2","6-1-2"])), "group"] = "rMS"
    data.loc[(data["prep_date"]=="13-04-2022") & (data["culture"].isin(["4-2-1","4-2-2","5-2-1","5-2-2","6-2-1","6-2-2"])), "group"] = "control"

    data.loc[(data["prep_date"]=="15-03-2022") & (data["culture"].isin(["5-2","6-2","7-2","9-2","9-4"])), "group"] = "rMS"
    data.loc[(data["prep_date"]=="15-03-2022") & (data["culture"].isin(["1-1","6-1","7-1","9-1","9-3"])), "group"] = "control"

    #data.loc[(data["prep_date"]=="07-04-2022") & (data["culture"].isin(["1-2-1","1-2-2","2-2-1","2-2-2","3-2-1","3-2-2"])), "group"] = "rMS"
    #data.loc[(data["prep_date"]=="07-04-2022") & (data["culture"].isin(["1-1-1","2-1-1","2-1-2","3-1-1","3-1-2"])), "group"] = "control"
    data.loc[(data["prep_date"]=="07-04-2022") & (data["culture"].isin(["1-2-1","1-2-2","2-2-1","2-2-2","3-2-1","3-2-2","4-2-1","4-2-2","5-2-1","5-2-2","9-2-1"])), "group"] = "rMS"
    data.loc[(data["prep_date"]=="07-04-2022") & (data["culture"].isin(["1-1-1","2-1-1","2-1-2","3-1-1","3-1-2","4-1-1","4-1-2","5-1-1","5-1-2","9-1-1","9-1-2"])), "group"] = "control"
    data =data.dropna()
    return data 

########## call function to get the final dataset ##############
# this step comes after the quality contorl of individual traces.
# after you decide which method to apply in the final dataset, use this function to gee the final dataset
def get_FR_with_different_methods(experiment):
    data = []
    for prep_date in results.keys():
        for group in results[prep_date].keys():
            for culture in results[prep_date][group].keys():
                series_raw = results[prep_date][group][culture]["time_series"]
                if len(series_raw)<2:
                    pass
                else:
                    series = np.array([rescale_and_detrend_trace(series_raw[i]) for i in series_raw.keys()])
                    cor = np.corrcoef(series)
                    np.fill_diagonal(cor,np.nan)

                    fs,Ss = get_power_spectrum(series_raw)
                    ave = np.mean(Ss,axis=0)
                    f=fs[0]

                    for i,neuron in enumerate(results[prep_date][group][culture]["time_series"].keys()):
                        raw_intensity = results[prep_date][group][culture]["time_series"][neuron]  
                        data.append(
                        {
                         "experiment": "iTBS",
                         "prep_date": prep_date,
                         "culture": culture[6:-4],
                         "timing": group,
                         "neuron": neuron,
                         "auc":np.abs(np.trapz(np.linspace(0,120,len(raw_intensity)),rescale_and_detrend_trace(raw_intensity))),
                         "firing_rate": analyze_trace(raw_intensity,"rescale_detrend",3.), # we used rescale_detrend_smooth\
                         "coef": np.nanmean(cor),
                         "power_auc": np.abs(np.trapz(np.linspace(0,fs[0][61],61),Ss[i][0:61])),
                        }
                        )

    df = pd.DataFrame(data)
    df = add_group_info(df)
    df.to_csv("calcium_data_"+experiment+".csv")


res_name = 'Hilus_itbs_data_res'
with open(res_name, 'rb') as f:
    results = pickle.load(f)
get_FR_with_different_methods("hilus")

res_name = 'DG_itbs_data_res'
with open(res_name, 'rb') as f:
    results = pickle.load(f)
get_FR_with_different_methods("DG")

