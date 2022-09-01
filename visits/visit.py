"""Creating a class for each instance in the data set i.e: one visit of one patient."""

import pandas as pd
import numpy as np
import glob
import ntpath
import scipy.ndimage as ndimage
import pymannkendall as mk
from datetime import datetime, timedelta
import logging
logger = logging.getLogger('ftpuploader')

def remove_outliers(df, n=3):
    mu = df['num_value'].mean()
    sigma = df['num_value'].std()

    df_no_outliers = df.copy()

    for idx, val in enumerate(df_no_outliers['num_value']):
        if not (mu - n*sigma < val < mu + 3*sigma):
            df_no_outliers['num_value'].iloc[idx] = mu

    return df_no_outliers

def residuals(df, sigma=100):
    df_smoothed = df.copy()
    df_residuals = df.copy()
    df_smoothed['num_value'] = ndimage.gaussian_filter1d(df['numvalue'], sigma=sigma)
    df_residuals['num_value'] = df['num_value'] - df_smoothed['num_value']
    return remove_outliers(df_residuals)

def rolling_mean(df, window=60):

    df['num_value'] = df['num_value'].fillna(df['num_value'].mean())

    k=0
    mean_df = pd.DataFrame(columns=['start_time','end_time','mean'])

    while (df['record_date_time'].iloc[0] + timedelta(minutes=window+k)) < (df['record_date_time'].iloc[-1]):
        index_1 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=k))
        index_2 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=window+k))

        while pd.isnull(df['num_value'][index_1]):
            index_1 += 1
        
        while pd.isnull(df['num_value'][index_2]):
            index_2 -= 1

        mean = df['num_value'][index_1:index_2].mean()

        mean_df = mean_df.append({'start_time': df['record_date_time'].iloc[index_1], 'end_time':df['record_date_time'].iloc[index_2], 'mean':mean}, ignore_index=True)
        k += 1
    
    # mean_df['mean'] = mean_df['mean'].fillna(mean_df['mean'].mean())
    mean_df['mean'].replace({pd.NaT: mean_df['mean'].mean()}, inplace = True)
    
    for i, row in enumerate(mean_df['mean']):
        if pd.isnull(row):
            df = df.drop(i)
    
    mean_df['mean'] = pd.to_numeric(mean_df['mean'])

    return mk.hamed_rao_modification_test(mean_df['mean'])


class Visit:
    def __init__(self, directory, visit_number, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details):
        self.directory = directory
        self.project_id = ntpath.basename(directory)
        self.cohort = None
        self.start_time = None
        self.end_time = None
        self.details = None
        self.age = None
        self.sex = None
        self.icu_type = None
        self.visit_no = visit_number
        # self.mean_trends = (rolling_mean(self.hr()))

        for id, v in zip(list(cohort_1_details["Project ID"]), list(cohort_1_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 1
                temp = cohort_1_details[cohort_1_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]
                self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        for id, v in zip(list(cohort_2_details["Project ID"]), list(cohort_2_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 2
                temp = cohort_2_details[cohort_2_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        for id, v in zip(list(cohort_3_details["Project ID"]), list(cohort_3_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 3
                temp = cohort_3_details[cohort_3_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        for id, v in zip(list(cohort_4_details["Project ID"]), list(cohort_4_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 4
                temp = cohort_4_details[cohort_4_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = self.details["failed_extubation_deid_date"].iloc[0]

        # if self.project_id in list(cohort_1_details["Project ID"]) and self.visit_no == :
        #     self.cohort = 1
        #     self.details = cohort_1_details[cohort_1_details["Project ID"] == self.project_id]
        #     self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        # if self.project_id in list(cohort_2_details["Project ID"]):
        #     self.cohort = 2
        #     self.details = cohort_2_details[cohort_2_details["Project ID"] == self.project_id]
        #     self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        # if self.project_id in list(cohort_3_details["Project ID"]):
        #     self.cohort = 3
        #     self.details = cohort_3_details[cohort_3_details["Project ID"] == self.project_id]
        #     self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        # if self.project_id in list(cohort_4_details["Project ID"]):
        #     self.cohort = 4
        #     self.details = cohort_4_details[cohort_4_details["Project ID"] == self.project_id]
        #     self.start_time = self.details["failed_extubation_deid_date"].iloc[0]


        if self.cohort == 1 or self.cohort == 4:
            self.end_time = self.details["re_intubation_deid_date"].iloc[0]
        if self.cohort == 3:
            self.end_time = self.details["death_deid_date"].iloc[0]
        # else:
        #     raise NotImplementedError


    
    def age(self):
        return ((self.details["extubation_deid_date"] - self.details["birth_deid_date"]).total_seconds())/86400

    def sex(self):
        return self.details["sex"]

    def icu_type(self):
        return self.details["icu_ward"]

    def hr(self):
        if self.cohort == 1 or self.cohort == 4 or self.cohort == 3:
            hr = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_HR") in file]
            hr_temp_df = pd.read_csv(hr[0], sep=",")
            hr_temp_df['record_date_time'] = pd.to_datetime(hr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            start_index = hr_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = hr_temp_df.record_date_time.searchsorted(self.end_time, side="right")
            hr_df = hr_temp_df.iloc[start_index+1:end_index-1, :]
            hr_df.index = range(len(hr_df))
            if not hr_df.empty:
                return hr_df
            else:
                print("There is no heart rate data available during the extubated period.")
                return None
        else:
            raise NotImplementedError("The psuedo extubation times have not been determined yet.")

    def rr(self):
        if self.cohort == 1 or self.cohort == 4 or self.cohort == 3:
            rr = [file for file in glob.glob(self.directory + "/*") if f"_{self.visit_no}_RR" in file]
            rr_temp_df = pd.read_csv(rr[0], sep=',')
            rr_temp_df['record_date_time'] = pd.to_datetime(rr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            start_index = rr_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = rr_temp_df.record_date_time.searchsorted(self.end_time)
            rr_df = rr_temp_df.iloc[start_index+1:end_index-1, :]
            for idx1 in rr_df.index:
                for idx2 in range(min(rr_df.index[-1], idx1+1), min(rr_df.index[-1], idx1+30), 1):
                    if rr_df['record_date_time'].loc[idx1] == rr_df['record_date_time'].loc[idx2]:
                        rr_df['num_value'].loc[idx2] = 0.5*(rr_df['num_value'].loc[idx1]+rr_df['num_value'].loc[idx2])
                        rr_df['monitor'].loc[idx2] = "combination"
                        rr_df = rr_df.drop(idx1, axis=0)
                        break
                    break
            rr_df.index = range(len(rr_df))
            if not rr_df.empty:
                return rr_df
            else:
                print("There is no respiration rate data available during the extubated period.")
                return None
        else:
            raise NotImplementedError("The psuedo extubation times have not been determined yet.")

    def abf(self):
        if self.cohort == 1 or self.cohort == 4 or self.cohort == 3:
            abf = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_ABF") in file]
            abf_temp_df = pd.read_csv(abf[0], sep=",")
            abf_temp_df['record_date_time'] = pd.to_datetime(abf_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            start_index = abf_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = abf_temp_df.record_date_time.searchsorted(self.end_time)
            abf_df = abf_temp_df.iloc[start_index+1:end_index-1, :]

            # abf_m_df = pd.DataFrame(columns=['monitor','record_date_time','num_value'])

            # for idx in abf_df.index:
            #     if 'm' in abf_df['monitor'].loc[idx]:
            #         abf_m_df = abf_m_df.append(abf_df.loc[idx])

            abf_m_df = abf_df[abf_df["monitor"].isin(["ABPm", "ARTm"])]

            abf_m_df.index = range(len(abf_m_df))
            if not abf_m_df.empty:
                return abf_m_df
            else:
                print("There is no blood pressure data available during the extubated period.")
                return None

    def mean_trends(self):
        return [rolling_mean(self.hr()).trend, rolling_mean(self.rr()).trend, rolling_mean(self.abf()).trend]
    

class Cohort:
    def __init__(self, list, n):
        self.n = n
        self.visits = list
    
    def mean_trend_count(self, hr=False, rr=False, abf=False):
        increasing, decreasing, no_trend, errors = 0, 0, 0, 0

        for visit in self.visits:
            try:
                if hr:
                    trend = rolling_mean(visit.hr())[0]
                    if trend == "increasing":
                        increasing += 1
                    if trend == "no trend":
                        no_trend += 1
                    if trend == "decreasing":
                        decreasing += 1
                
                if rr:
                    trend = rolling_mean(visit.rr())[0]
                    if trend == "increasing":
                        increasing += 1
                    if trend == "no trend":
                        no_trend += 1
                    if trend == "decreasing":
                        decreasing += 1
                
                if abf:
                    trend = rolling_mean(visit.abf())[0]
                    if trend == "increasing":
                        increasing += 1
                    if trend == "no trend":
                        no_trend += 1
                    if trend == "decreasing":
                        decreasing += 1
            except Exception as e:
                errors += 1
                logger.error(visit.project_id + ' visit ' + str(visit.visit_no) + ' failed: '+ str(e))

        
        return (increasing, no_trend, decreasing, errors)