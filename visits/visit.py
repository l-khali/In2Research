"""Creating a class for each instance in the data set i.e: one visit of one patient."""

import pandas as pd
import numpy as np
import glob
import ntpath

class Visit:
    def __init__(self, directory, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details):
        self.directory = directory
        self.project_id = ntpath.basename(directory)
        self.cohort = None
        self.start_time = None
        self.end_time = None
        self.details = None
        self.age = None
        self.sex = None
        self.icu_type = None

        if self.project_id in list(cohort_1_details["Project ID"]):
            self.cohort = 1
            self.details = cohort_1_details[cohort_1_details["Project ID"] == self.project_id]
            self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        if self.project_id in list(cohort_2_details["Project ID"]):
            self.cohort = 2
            self.details = cohort_2_details[cohort_2_details["Project ID"] == self.project_id]
            self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        if self.project_id in list(cohort_3_details["Project ID"]):
            self.cohort = 3
            self.details = cohort_3_details[cohort_3_details["Project ID"] == self.project_id]
            self.start_time = self.details["failed_extubation_deid_date"].iloc[0]
        if self.project_id in list(cohort_4_details["Project ID"]):
            self.cohort = 4
            self.details = cohort_4_details[cohort_4_details["Project ID"] == self.project_id]
            self.start_time = self.details["failed_extubation_deid_date"].iloc[0]


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
            hr = [file for file in glob.glob(self.directory + "/*") if "_HR" in file]
            print(hr)
            hr_temp_df = pd.read_csv(hr[0], sep=",")
            hr_temp_df['record_date_time'] = pd.to_datetime(hr_temp_df['record_date_time'], format='%Y-%m-%d %H:%M:%S')
            start_index = hr_temp_df.record_date_time.searchsorted(self.start_time)
            end_index = hr_temp_df.record_date_time.searchsorted(self.end_time)
            hr_df = hr_temp_df.iloc[start_index+1:end_index-1, :]
            return hr_df
        # else:
        #     raise NotImplementedError