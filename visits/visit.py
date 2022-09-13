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
        if not (mu - n*sigma < val < mu + n*sigma):
            df_no_outliers['num_value'].iloc[idx] = mu

    return df_no_outliers

def residuals(df, sigma=100):
    df_smoothed = df.copy()
    df_residuals = df.copy()
    df_smoothed['num_value'] = ndimage.gaussian_filter1d(df['num_value'], sigma=sigma)
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

def rolling_variance(df, window=60):

    df = residuals(df)

    df['num_value'] = df['num_value'].fillna(df['num_value'].mean())

    k=0
    var_df = pd.DataFrame(columns=['start_time','end_time','variance'])

    while (df['record_date_time'].iloc[0] + timedelta(minutes=window+k)) < (df['record_date_time'].iloc[-1]):
        index_1 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=k))
        index_2 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=window+k))

        while pd.isnull(df['num_value'][index_1]):
            index_1 += 1
        
        while pd.isnull(df['num_value'][index_2]):
            index_2 -= 1

        var = df['num_value'][index_1:index_2].var()

        var_df = var_df.append({'start_time': df['record_date_time'].iloc[index_1], 'end_time':df['record_date_time'].iloc[index_2], 'variance':var}, ignore_index=True)
        k += 1
    
    # mean_df['mean'] = mean_df['mean'].fillna(mean_df['mean'].mean())
    var_df['variance'].replace({pd.NaT: var_df['variance'].mean()}, inplace = True)
    
    for i, row in enumerate(var_df['variance']):
        if pd.isnull(row):
            df = df.drop(i)
    
    var_df['variance'] = pd.to_numeric(var_df['variance'])

    return mk.hamed_rao_modification_test(var_df['variance'])

def rolling_autocorrelation(df, window=60):

    df = residuals(df)

    df['num_value'] = df['num_value'].fillna(df['num_value'].mean())

    k=0
    ac_df = pd.DataFrame(columns=['start_time','end_time','autocorrelation'])

    while (df['record_date_time'].iloc[0] + timedelta(minutes=window+k)) < (df['record_date_time'].iloc[-1]):
        index_1 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=k))
        index_2 = df.record_date_time.searchsorted(df['record_date_time'].iloc[0] + timedelta(minutes=window+k))

        while pd.isnull(df['num_value'][index_1]):
            index_1 += 1
        
        while pd.isnull(df['num_value'][index_2]):
            index_2 -= 1

        ac = df['num_value'][index_1:index_2].autocorr()

        ac_df = ac_df.append({'start_time': df['record_date_time'].iloc[index_1], 'end_time':df['record_date_time'].iloc[index_2], 'autocorrelation':ac}, ignore_index=True)
        k += 1
    
    # mean_df['mean'] = mean_df['mean'].fillna(mean_df['mean'].mean())
    ac_df['autocorrelation'].replace({pd.NaT: ac_df['autocorrelation'].mean()}, inplace = True)
    
    for i, row in enumerate(ac_df['autocorrelation']):
        if pd.isnull(row):
            df = df.drop(i)
    
    ac_df['autocorrelation'] = pd.to_numeric(ac_df['autocorrelation'])

    return mk.hamed_rao_modification_test(ac_df['autocorrelation'])

def holm_bonferroni(project_ids, visit_no, p_vals, trends, alpha = 0.05):
    results = np.array([[id, v, "{:f}".format(p), t] for id, v, p, t in zip(project_ids, visit_no, p_vals, trends)])
    n = len(p_vals)
    results = results[results[:, 2].argsort()]

    id_sorted = results[:,0]
    visits_sorted = results[:,1]
    p_vals_sorted = results[:,2]
    trends_sorted = results[:,3]
    alphas = [float(alpha/(n-rank+1)) for rank in range(1,n+1)]

    failed = []

    for i, id, v, p, trend, alpha in zip(range(1, n+1), id_sorted, visits_sorted, p_vals_sorted, trends_sorted, alphas):
        if float(p) < alpha and trend == "increasing":
            # print(f"Test {i} passed")
            pass
        else:
            failed.append((i, id, v, p, trend))
            print(f"Test {i} of {n} failed: ({p},{trend})")
    
    if failed:
        return failed
    else:
        return "All tests passed"

def icu_split(results_df, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details):
    # final = pd.DataFrame(columns=["measure","biological measure","","icu_ward", "total", "failed", "proportion"])
    # final["measure"] = [measure, measure, measure]
    # final["biological measure"] = [bio_measure, bio_measure, bio_measure]
    # final["icu_ward"] = ["PICU", "FLAMI", "NICU"]

    picu_results_df = pd.DataFrame(columns=["trend", "p-value", "tau", "Project ID", "visit_no"])
    flami_results_df = pd.DataFrame(columns=["trend", "p-value", "tau", "Project ID", "visit_no"])
    nicu_results_df = pd.DataFrame(columns=["trend", "p-value", "tau", "Project ID", "visit_no"])

    for idx, project_id, visit_no in zip(results_df.index, results_df["Project ID"], results_df["visit_no"]):
        v = Visit(f"../In2Research_data/data/{project_id}", visit_no, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details)
        ward = v.icu_type()
        if ward.iloc[0] == "PICU":
            picu_results_df = picu_results_df.append(results_df.loc[idx])
        if ward.iloc[0] == "FLAMI":
            flami_results_df = flami_results_df.append(results_df.loc[idx])
        if ward.iloc[0] == "NICU":
            nicu_results_df = nicu_results_df.append(results_df.loc[idx])
    
    results_list = [picu_results_df, flami_results_df, nicu_results_df]
    failed_list = [holm_bonferroni(df["Project ID"], df["visit_no"], df["p-value"], df["trend"], alpha = 0.05) for df in results_list]
    pass_proportions = [(1 - (len(x)/len(y))) for x, y in zip(failed_list, results_list)]
    
    return pass_proportions, results_list, failed_list


class Visit:
    def __init__(self, directory, visit_number, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details):
        self.directory = directory
        self.project_id = ntpath.basename(directory)
        self.cohort = None
        self.start_time = None
        self.end_time = None
        self.details = None
        # self.age = None
        # self.sex = None
        # self.icu_type = None
        self.visit_no = visit_number
        # self.final_datapoint = None
        # self.hr_endpoint = None
        # self.rr_endpoint = None
        # self.abf_endpoint = None
        # self.mean_trends = (rolling_mean(self.hr()))

        for id, v in zip(list(cohort_1_details["Project ID"]), list(cohort_1_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 1
                temp = cohort_1_details[cohort_1_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]
                self.start_time = pd.to_datetime(self.details["failed_extubation_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        for id, v in zip(list(cohort_2_details["Project ID"]), list(cohort_2_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 2
                temp = cohort_2_details[cohort_2_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = pd.to_datetime(self.details["extubation_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        for id, v in zip(list(cohort_3_details["Project ID"]), list(cohort_3_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 3
                temp = cohort_3_details[cohort_3_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = pd.to_datetime(self.details["extubation_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        for id, v in zip(list(cohort_4_details["Project ID"]), list(cohort_4_details["icu_visit"])):
            if id == self.project_id and v == self.visit_no:
                self.cohort = 4
                temp = cohort_4_details[cohort_4_details["Project ID"] == self.project_id]
                self.details = temp[temp["icu_visit"] == self.visit_no]                
                self.start_time = pd.to_datetime(self.details["failed_extubation_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')

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
            self.end_time = pd.to_datetime(self.details["re_intubation_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        if self.cohort == 3:
            self.end_time = pd.to_datetime(self.details["death_deid_date"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        if self.cohort == 2:
            self.end_time = pd.to_datetime(self.details["pseudo_reintubation"].iloc[0], format='%Y-%m-%d %H:%M:%S')
        
        # else:
        #     raise NotImplementedError


    
    def age(self):
        return ((self.details["extubation_deid_date"] - self.details["birth_deid_date"]).total_seconds())/86400

    def sex(self):
        return self.details["sex"]

    def icu_type(self):
        return self.details["icu_ward"]

    def hr(self):
        if self.cohort:
            hr = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_HR") in file]
            hr_temp_df = pd.read_csv(hr[0], sep=",")
            hr_temp_df['record_date_time'] = pd.to_datetime(hr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            # self.hr_endpoint = hr_temp_df['record_date_time'].iloc[-1]
            start_index = hr_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = hr_temp_df.record_date_time.searchsorted(self.end_time)

            # IF DURATION OF THE DATA INSIDE EXTUBATED TIME IS LESS THAN 120 MIN THEN RAISE ERROR

            if (self.end_time - self.start_time).seconds/60 < 120:
                raise TooShortError("Extubated period is too short to carry out analysis.")

            if (self.end_time - self.start_time).seconds/60 > 900:
                raise TooShortError("Extubated period is too short to carry out analysis.")
            # if (self.end_time - hr_temp_df['record_date_time'].iloc[end_index-1]).seconds/60 > 20:
            #     raise MissingDataError("Time series data is not available close enough to the critical point")

            hr_df = hr_temp_df.iloc[start_index+1:end_index-1, :]
            hr_df.index = range(len(hr_df))
            if not hr_df.empty:
                return hr_df
            else:
                print("There is no heart rate data available during the extubated period.")
                return None
        # if self.cohort == 2:
        #     hr = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_HR") in file]
        #     hr_temp_df = pd.read_csv(hr[0], sep=",")
        #     hr_temp_df['record_date_time'] = pd.to_datetime(hr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
        #     if not hr_temp_df.empty:
        #         self.hr_endpoint = hr_temp_df['record_date_time'].iloc[-1]
        #     else:
        #         print("There is no blood pressure data available during the extubated period.")
        #         return None

    def hr_endpoint(self):
        hr = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_HR") in file]
        if hr:
            hr_temp_df = pd.read_csv(hr[0], sep=",")
            hr_temp_df['record_date_time'] = pd.to_datetime(hr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            if not hr_temp_df.empty:
                return hr_temp_df['record_date_time'].iloc[-1]
            else:
                print("There is no heart rate data available during the extubated period.")
                return None
        else:
            return None

    def rr(self):
        if self.cohort:
            rr = [file for file in glob.glob(self.directory + "/*") if f"_{self.visit_no}_RR" in file]
            rr_temp_df = pd.read_csv(rr[0], sep=',')
            rr_temp_df['record_date_time'] = pd.to_datetime(rr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            # self.rr_endpoint = rr_temp_df['record_date_time'].iloc[-1]
            start_index = rr_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = rr_temp_df.record_date_time.searchsorted(self.end_time)

            # if (self.end_time - rr_temp_df['record_date_time'].iloc[end_index-1]).seconds/60 > 20:
            #     raise MissingDataError("Time series data is not available close enough to the critical point")

            rr_df = rr_temp_df.iloc[start_index+1:end_index-1, :]
            for idx1 in rr_df.index:
                for idx2 in range(min(rr_df.index[-1], idx1+1), min(rr_df.index[-1], idx1+3), 1):
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
        # if self.cohort == 2:
        #     rr = [file for file in glob.glob(self.directory + "/*") if f"_{self.visit_no}_RR" in file]
        #     rr_temp_df = pd.read_csv(rr[0], sep=',')
        #     rr_temp_df['record_date_time'] = pd.to_datetime(rr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
        #     if not rr_temp_df.empty:
        #         self.rr_endpoint = rr_temp_df['record_date_time'].iloc[-1]
        #     else:
        #         print("There is no blood pressure data available during the extubated period.")
        #         return None
            # raise NotImplementedError("The psuedo extubation times have not been determined yet.")

    def rr_endpoint(self):
        rr = [file for file in glob.glob(self.directory + "/*") if f"_{self.visit_no}_RR" in file]
        if rr:
            rr_temp_df = pd.read_csv(rr[0], sep=',')
            rr_temp_df['record_date_time'] = pd.to_datetime(rr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            if not rr_temp_df.empty:
                return rr_temp_df['record_date_time'].iloc[-1]
            else:
                print("There is no blood pressure data available during the extubated period.")
                return None
        else:
            return None

    def abf(self):
        if self.cohort:
            abf = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_ABF") in file]
            abf_temp_df = pd.read_csv(abf[0], sep=",")
            abf_temp_df['record_date_time'] = pd.to_datetime(abf_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            # self.abf_endpoint = abf_temp_df['record_date_time'].iloc[-1]
            start_index = abf_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = abf_temp_df.record_date_time.searchsorted(self.end_time)

            # if (self.end_time - abf_temp_df['record_date_time'].iloc[end_index-1]).seconds/60 > 20:
            #     raise MissingDataError("Time series data is not available close enough to the critical point")

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
        # if self.cohort == 2:
        #     abf = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_ABF") in file]
        #     abf_temp_df = pd.read_csv(abf[0], sep=",")
        #     abf_temp_df['record_date_time'] = pd.to_datetime(abf_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
        #     if not abf_temp_df.empty:
        #         self.abf_endpoint = abf_temp_df['record_date_time'].iloc[-1]
        #     else:
        #         print("There is no blood pressure data available during the extubated period.")
        #         return None
            # raise NotImplementedError

    def abf_endpoint(self):
        abf = [file for file in glob.glob(self.directory + "/*") if (f"_{self.visit_no}_ABF") in file]
        if abf:
            abf_temp_df = pd.read_csv(abf[0], sep=",")
            abf_temp_df['record_date_time'] = pd.to_datetime(abf_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            if not abf_temp_df.empty:
                return abf_temp_df['record_date_time'].iloc[-1]
            else:
                print("There is no blood pressure data available during the extubated period.")
                return None
        else:
            return None

    def mean_trends(self):
        return [rolling_mean(self.hr()).trend, rolling_mean(self.rr()).trend, rolling_mean(self.abf()).trend]
    
    def final_datapoint(self):
        """Find latest final data point for a given instance."""

        # try:
        return max(x for x in [self.hr_endpoint(), self.rr_endpoint(), self.abf_endpoint()] if x)
        # except Exception as e:
        #         logger.error(str(e))
    

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
                print((increasing, no_trend, decreasing, errors))

        return (increasing, no_trend, decreasing, errors)

    def var_results(self, hr=False, rr=False, abf=False):
        # increasing, decreasing, no_trend, errors = 0, 0, 0, 0
        errors = 0
        result_df = pd.DataFrame(columns=['Project ID', 'visit_no', 'trend', "p-value", "tau"])

        for visit in self.visits:
            try:
                if hr:
                    mk = rolling_variance(visit.hr())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': float(mk[2]), 'tau': float(mk[4])}, ignore_index=True)

                if rr:
                    mk = rolling_variance(visit.rr())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': float(mk[2]), 'tau': float(mk[4])}, ignore_index=True)
                
                if abf:
                    mk = rolling_variance(visit.abf())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': float(mk[2]), 'tau': float(mk[4])}, ignore_index=True)
            except Exception as e:
                errors += 1
                logger.error(visit.project_id + ' visit ' + str(visit.visit_no) + ' failed: '+ str(e))
                # print((increasing, no_trend, decreasing, errors))

        return (result_df, f"Missing data/errors: {errors}")
    
    def ac_results(self, hr=False, rr=False, abf=False):
        # increasing, decreasing, no_trend, errors = 0, 0, 0, 0
        errors = 0
        result_df = pd.DataFrame(columns=['trend', "p-value", "tau"])

        for visit in self.visits:
            try:
                if hr:
                    mk = rolling_autocorrelation(visit.hr())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': mk[2], 'tau': mk[4]}, ignore_index=True)

                if rr:
                    mk = rolling_autocorrelation(visit.rr())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': mk[2], 'tau': mk[4]}, ignore_index=True)
                
                if abf:
                    mk = rolling_autocorrelation(visit.abf())
                    result_df = result_df.append({'Project ID': visit.project_id, 'visit_no': visit.visit_no, 'trend': mk[0], 'p-value': mk[2], 'tau': mk[4]}, ignore_index=True)
            except Exception as e:
                errors += 1
                logger.error(visit.project_id + ' visit ' + str(visit.visit_no) + ' failed: '+ str(e))
                # print((increasing, no_trend, decreasing, errors))

        return (result_df, f"Missing data/errors: {errors}")


class MissingDataError(Exception):
    """Exception raised if the time series does not contain enough data for effective analysis."""

    pass


class TooShortError(Exception):
    """Exception raised if the time series does not contain enough data for effective analysis."""

    pass
