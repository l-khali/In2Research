"""Creating a class for each instance in the data set i.e: one visit of one patient."""

import pandas as pd
import numpy as np
import glob
import ntpath

class Visit:
    def __init__(self, directory, cohort_1_details, cohort_2_details, cohort_3_details, cohort_4_details = None):
        self.directory = directory
        self.project_id = ntpath.basename(directory)
        self.cohort = None
        self.start_time = None
        self.end_time = None
        self.details = None

        if self.project_id in list(cohort_1_details["Project ID"]):
            self.cohort = 1
            self.details = cohort_1_details[cohort_1_details["Project ID"] == self.project_id]
        if self.project_id in list(cohort_2_details["Project ID"]):
            self.cohort = 2
            self.details = cohort_2_details[cohort_2_details["Project ID"] == self.project_id]
        if self.project_id in list(cohort_3_details["Project ID"]):
            self.cohort = 3
            self.details = cohort_3_details[cohort_3_details["Project ID"] == self.project_id]
        if cohort_4_details and (self.project_id in list(cohort_4_details["Project ID"])):
            self.cohort = 4
            self.details = cohort_4_details[cohort_4_details["Project ID"] == self.project_id]


    def hr(self):
        raise NotImplementedError