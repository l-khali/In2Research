"""Creating a class for each instance in the data set i.e: one visit of one patient."""

import pandas as pd
import numpy as np
import glob
import ntpath

class Visit:
    def __init__(self, directory):
        self.directory = directory
        self.project_id = ntpath.basename(directory)
        