"""Creating a class for each instance in the data set i.e: one visit of one patient."""

import pandas as pd
import numpy as np
import glob
import ntpath

def remove_outliers(df, n=3):
    mu = df['num_value'].mean()
    sigma = df['num_value'].std()

    df_no_outliers = df.copy()

    for idx, val in enumerate(df_no_outliers['num_value']):
        if not (mu - n*sigma < val < mu + 3*sigma):
            df_no_outliers['num_value'].iloc[idx] = mu

    return df_no_outliers


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
            end_index = hr_temp_df.record_date_time.searchsorted(self.end_time)
            hr_df = hr_temp_df.iloc[start_index+1:end_index-1, :]
            return remove_outliers(hr_df)
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
            return remove_outliers(rr_df)
        else:
            raise NotImplementedError("The psuedo extubation times have not been determined yet.")