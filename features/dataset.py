from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.factory import FeatureExtractionResult

import mne
import matplotlib.pyplot as plt
import numpy as np


class FeatureDataset:
    def __init__(self, features_df:pd.DataFrame, ppc_raw_data:np.ndarray, subject_dico:dict, pipeline_name:str, eeg_info):
        self.features_df = features_df
        self.ppc_raw_data = ppc_raw_data #(n_channels, n_channels, n_bands)
        self.subject_dico = subject_dico
        self.pipeline_name = pipeline_name

    def plot(self, feature_name:str, title:str=None, sub_title:str=None, figsize=(7,6), contours=7, cmap="RdBu_r"):
        return 0






class FeatureDatasetFactory:
    @staticmethod
    def from_extraction_result(extraction_result:FeatureExtractionResult):
        features_df = extraction_result.dataframe
        ppc_raw_data = extraction_result.ppc_result.dense_data_raw
        subject_dico = extraction_result.eeg.source.subject.to_dict()
        pipeline_name = extraction_result.eeg.pipeline_name
        return FeatureDataset(features_df, ppc_raw_data, subject_dico, pipeline_name)