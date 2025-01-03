import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 

#define gather functions
def data_gather(cultureID,batch_name,groupIDs,groupnames):
    #relocate to each subfolder
    T = T_general + str(cultureID) + "/"
    #get dataset and make a copy to work
    data_raw  = pd.read_csv(T+"Results_area.csv")
    data      = data_raw.copy()
    #delete unwanted columns
    data.pop("Mean")
    data.pop("Min")
    data.pop("Max")
    data.pop("IntDen")
    data.pop("RawIntDen")
    #enter area size
    area_CA1  = data[data["Label"]=="middle_ch02-CA1-raw.tif"]["Area"]
    area_CA3  = data[data["Label"]=="middle_ch02-CA3-raw.tif"]["Area"]
    area_EC   = data[data["Label"]=="middle_ch02-EC-raw.tif"]["Area"]
    area_DG   = data[data["Label"]=="middle_ch02-DG-raw.tif"]["Area"]
    #enter region name to each entry
    data['region'] = [0]*len(data)
    data.loc[data["Label"]=="middle_ch02-CA1-raw.tif","region"] = 'CA1'
    data.loc[data["Label"]=="middle_ch02-CA3-raw.tif","region"] = 'CA3'
    data.loc[data["Label"]=="middle_ch02-DG-raw.tif","region"]  = 'DG'
    data.loc[data["Label"]=="middle_ch02-EC-raw.tif","region"]  = 'EC'
    #fetch elements in the culture ID to name individual dataset
    elements = [i for i in cultureID]
    data["group_ID"] = [elements[0]]*len(data)
    data["culture_ID"] = [elements[2]]*len(data)
    for i in np.arange(0,len(groupIDs),1):
    	if elements[0]==str(groupIDs[i]):
    		data["group"] = [groupnames[i]]*len(data)
    count_CA1 = len(pd.read_csv(T+"count_CA1.csv"))
    count_CA3 = len(pd.read_csv(T+"count_CA3.csv"))
    count_EC  = len(pd.read_csv(T+"count_EC.csv"))
    count_DG  = len(pd.read_csv(T+"count_DG.csv"))
    #enter cell count data to this dataframe
    data['count'] = [0]*len(data)
    data.loc[data["Label"]=="middle_ch02-CA1-raw.tif","count"] = np.array([count_CA1])
    data.loc[data["Label"]=="middle_ch02-CA3-raw.tif","count"] = np.array([count_CA3])
    data.loc[data["Label"]=="middle_ch02-DG-raw.tif","count"] = np.array([count_DG])
    data.loc[data["Label"]=="middle_ch02-EC-raw.tif","count"] = np.array([count_EC])
    data['batch_name'] = [batch_name]*len(data)
    return data


#define a second gather function only because the channel was named differently
def data_gather2(cultureID,batch_name,groupIDs,groupnames):
    #relocate to each subfolder
    T = T_general + str(cultureID) + "/"
    #get dataset and make a copy to work
    data_raw  = pd.read_csv(T+"Results_area.csv")
    data      = data_raw.copy()
    #delete unwanted columns
    data.pop("Mean")
    data.pop("Min")
    data.pop("Max")
    data.pop("IntDen")
    data.pop("RawIntDen")
    #enter area size
    area_CA1  = data[data["Label"]=="middle_ch01-CA1-raw.tif"]["Area"]
    area_CA3  = data[data["Label"]=="middle_ch01-CA3-raw.tif"]["Area"]
    area_EC   = data[data["Label"]=="middle_ch01-EC-raw.tif"]["Area"]
    area_DG   = data[data["Label"]=="middle_ch01-DG-raw.tif"]["Area"]
    #enter region name to each entry
    data['region'] = [0]*len(data)
    data.loc[data["Label"]=="middle_ch01-CA1-raw.tif","region"] = 'CA1'
    data.loc[data["Label"]=="middle_ch01-CA3-raw.tif","region"] = 'CA3'
    data.loc[data["Label"]=="middle_ch01-DG-raw.tif","region"]  = 'DG'
    data.loc[data["Label"]=="middle_ch01-EC-raw.tif","region"]  = 'EC'
    #fetch elements in the culture ID to name individual dataset
    elements = [i for i in cultureID]
    data["group_ID"] = [elements[0]]*len(data)
    data["culture_ID"] = [elements[2]]*len(data)
    for i in np.arange(0,len(groupIDs),1):
    	if elements[0]==str(groupIDs[i]):
    		data["group"] = [groupnames[i]]*len(data)
    count_CA1 = len(pd.read_csv(T+"count_CA1.csv"))
    count_CA3 = len(pd.read_csv(T+"count_CA3.csv"))
    count_EC  = len(pd.read_csv(T+"count_EC.csv"))
    count_DG  = len(pd.read_csv(T+"count_DG.csv"))
    #enter cell count data to this dataframe
    data['count'] = [0]*len(data)
    data.loc[data["Label"]=="middle_ch01-CA1-raw.tif","count"] = np.array([count_CA1])
    data.loc[data["Label"]=="middle_ch01-CA3-raw.tif","count"] = np.array([count_CA3])
    data.loc[data["Label"]=="middle_ch01-DG-raw.tif","count"] = np.array([count_DG])
    data.loc[data["Label"]=="middle_ch01-EC-raw.tif","count"] = np.array([count_EC])
    data['batch_name'] = [batch_name]*len(data)
    return data



#define a normalize function
def normalized_data_by_sham(data):
    data_new = data[data["Label"]==0]
    for batch in np.unique(data["batch_name"]):
        data_batch = data[data["batch_name"]==batch]
        for region in np.unique(data_batch["region"]):
            data_region = data_batch[data_batch["region"]==region]
            if batch == "2911":
                sham_data        = data_region[data_region["group"]=="sham"]
                rMS_data         = data_region[data_region["group"]=="rMS"]
                sham_data_normed = sham_data["count"].values/np.mean(sham_data["count"].values)
                rMS_data_normed  = rMS_data["count"].values/np.mean(sham_data["count"].values)

                sham_data.loc[sham_data["batch_name"]==batch,"norm_count_by_sham"] = sham_data_normed
                rMS_data.loc[rMS_data["batch_name"]==batch,"norm_count_by_sham"] = rMS_data_normed
                data_new = pd.concat([data_new,sham_data])
                data_new = pd.concat([data_new,rMS_data])

                sham_24h_data        = data_region[data_region["group"]=="sham-24h"]
                rMS_24h_data         = data_region[data_region["group"]=="24h"]
                sham_24h_data_normed = sham_24h_data["count"]/np.mean(sham_24h_data["count"].values)
                rMS_24h_data_normed  = rMS_24h_data["count"]/np.mean(sham_24h_data["count"].values)

                sham_24h_data.loc[sham_24h_data["batch_name"]==batch,"norm_count_by_sham"] = sham_24h_data_normed
                rMS_24h_data.loc[rMS_24h_data["batch_name"]==batch,"norm_count_by_sham"] = rMS_24h_data_normed
                data_new = pd.concat([data_new,sham_24h_data])
                data_new = pd.concat([data_new,rMS_24h_data])

            if batch == "1903":
                sham_24h_data        = data_region[data_region["group"]=="sham-24h"]
                rMS_24h_data         = data_region[data_region["group"]=="24h"]
                sham_24h_data_normed = sham_24h_data["count"]/np.mean(sham_24h_data["count"].values)
                rMS_24h_data_normed  = rMS_24h_data["count"]/np.mean(sham_24h_data["count"].values)
                sham_24h_data.loc[sham_24h_data["batch_name"]==batch,"norm_count_by_sham"] = sham_24h_data_normed
                rMS_24h_data.loc[rMS_24h_data["batch_name"]==batch,"norm_count_by_sham"] = rMS_24h_data_normed
                data_new = pd.concat([data_new,sham_24h_data])
                data_new = pd.concat([data_new,rMS_24h_data])

            if batch == "1101":
                sham_data        = data_region[data_region["group"]=="sham"]
                rMS_data         = data_region[data_region["group"]=="rMS"]
                sham_data_normed = sham_data["count"]/np.mean(sham_data["count"].values)
                rMS_data_normed  = rMS_data["count"]/np.mean(sham_data["count"].values)
                sham_data.loc[sham_data["batch_name"]==batch,"norm_count_by_sham"] = sham_data_normed
                rMS_data.loc[rMS_data["batch_name"]==batch,"norm_count_by_sham"] = rMS_data_normed
                data_new = pd.concat([data_new,sham_data])
                data_new = pd.concat([data_new,rMS_data])
    return data_new


############### load 1st batch ##############
T_general = './raw_images/2911-cfos/'
batch_name = '2911'
groupIDs   = [1,2,6,7]
groupnames = ["sham","rMS","sham-24h","24h"]

data_merged = data_gather("1-1",batch_name,groupIDs,groupnames)
for cultureID in ["1-2","1-3","1-4","2-1","2-2","2-3","2-4","6-1","6-2","6-3","6-4","7-1","7-2","7-3","7-4"]:
    data_each   = data_gather(cultureID,batch_name,groupIDs,groupnames)
    data_merged = pd.concat((data_merged,data_each))


############### load 2nd batch ##############
T_general = './raw_images/1903-cfos/'
batch_name = '1903'
groupIDs   = [0,1]
groupnames = ["sham-24h","24h"]

for cultureID in ["0-1"]:
    data_each   = data_gather(cultureID,batch_name,groupIDs,groupnames)
    data_merged = pd.concat((data_merged,data_each))

for cultureID in ["0-2","0-3","0-4","0-5","0-6","1-1","1-2","1-3","1-4","1-5","1-6"]:
    data_each   = data_gather2(cultureID,batch_name,groupIDs,groupnames)
    data_merged = pd.concat((data_merged,data_each))

#########load 3rd batch##########
T_general = './raw_images/1101-cfos/'
batch_name = '1101'
groupIDs   = [1,3]
groupnames = ["sham","rMS"]

for cultureID in ["1-1","1-2","1-3","1-4","3-1","3-2","3-3","3-4"]:
    data_each   = data_gather(cultureID,batch_name,groupIDs,groupnames)
    data_merged = pd.concat((data_merged,data_each))

data = normalized_data_by_sham(data_merged)
data.to_csv('90min_24h_cfos_count.csv', index=False)