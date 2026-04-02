from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.factory import FeatureExtractionResult
from participants.definition import ParticipantFactory

import mne
import matplotlib.pyplot as plt
import numpy as np

from features.visualization import TopomapFactory





class SingleParticipantProcessedFeatureDataset:
    def __init__(self, features_df:pd.DataFrame, ppc_raw_data:np.ndarray, subject_dico:dict, pipeline_name:str, eeg_info_dico:dict):
        self.features_df = features_df
        self.ppc_raw_data = ppc_raw_data #(n_channels, n_channels, n_bands)
        self.subject_dico = subject_dico
        self.subject = ParticipantFactory.build(subject_dico)
        self.pipeline_name = pipeline_name
        self.eeg_info_dico = eeg_info_dico
        self.eeg_info = mne.Info.from_json_dict(eeg_info_dico)

    def plot(self, feature_name:str, title:str=None, sub_title:str=None, figsize=(7,6), contours=7, cmap="RdBu_r"):

        

        info = self.eeg_info
        values = self.features_df[feature_name].values
        vlim = (min(values), max(values))

        
        figure_title = title if title else feature_name
        subject = self.subject
        subject_description = f"Subject {subject.id} : {subject.health_state} | MMSE : {subject.mmse} | age : {subject.age}Y | gender : {subject.gender}"

        figure_subtitle = sub_title if sub_title else subject_description

        

        TopomapFactory.plot(values, info, figure_title, figure_subtitle, figsize, contours, cmap, vlim, sensors=True)

    @property
    def feature_names(self):
        return list(self.features_df.columns)
    
    @property
    def ch_names(self):
        return list(self.features_df.index)





class SingleParticipantProcessedFeatureDatasetFactory:
    @staticmethod
    def build(extraction_result:FeatureExtractionResult):
        features_df = extraction_result.dataframe
        ppc_raw_data = extraction_result.ppc_result.dense_data_raw
        subject_dico = extraction_result.eeg.source.subject.to_dict()
        pipeline_name = extraction_result.eeg.pipeline_name
        eeg_info_dico = extraction_result.eeg.info.to_json_dict()
        return SingleParticipantProcessedFeatureDataset(features_df, ppc_raw_data, subject_dico, pipeline_name, eeg_info_dico)
    



from participants.groups import HealthState
from utils.enum import EnumParser
from utils.dataframe import DataframeHelpers


class SampleSelector:

    def __init__(self, dataset: "FeaturesDataset"):
        self.dataset = dataset
        self._long_df_cache = None

    @property
    def long_df(self):
        if self._long_df_cache is None:
            self._long_df_cache = self.dataset.to_long_dataframe()
        return self._long_df_cache

    def select_feature(self, feature: str) -> pd.DataFrame:

        return self.long_df[self.long_df["feature"] == feature]

    def select_channel(self, feature: str, channel: str):

        df = self.select_feature(feature)
        return df[df["channel"] == channel]

    def select_groups(
        self,
        feature: str,
        group_col: str,
        group_a: str,
        group_b: str,
        channel: str | None = None,
        value_col: str = "value",
    ):

        df = self.select_feature(feature)

        if channel:
            df = df[df["channel"] == channel]

        group_a_df = df[df[group_col] == group_a][value_col]
        group_b_df = df[df[group_col] == group_b][value_col]

        return group_a_df, group_b_df
    
    def select_groups_all_channels(
        self,
        feature,
        group_col,
        group_a,
        group_b,
        value_col="value"
    ):

        df = self.select_feature(feature)

        grouped = {}

        for channel in self.dataset.ch_names:

            channel_df = df[df["channel"] == channel]

            group_a_vals = channel_df[channel_df[group_col] == group_a][value_col]
            group_b_vals = channel_df[channel_df[group_col] == group_b][value_col]

            if not group_a_vals.empty and not group_b_vals.empty:

                grouped[channel] = (group_a_vals, group_b_vals)

        return grouped


class FeaturesDataset:
    def __init__(self, participant_datasets:list[SingleParticipantProcessedFeatureDataset]):
        self.participant_datasets = participant_datasets

    @property
    def subjects(self):
        return [dataset.subject for dataset in self.participant_datasets]
    
    @property
    def ch_names(self) :
        return self.participant_datasets[0].ch_names
    
    @property
    def feature_names(self):
        return self.participant_datasets[0].feature_names
    
    @property
    def groups(self):
        return set([subject.group for subject in self.subjects])
    
    @property
    def eeg_info(self):
        return self.participant_datasets[0].eeg_info
    
    @property
    def pipeline_name(self):
        return self.participant_datasets[0].pipeline_name
    
    def particpant_dataset(self, participant_id:str):
        for dataset in self.participant_datasets :
            if dataset.subject.id == participant_id :
                return dataset
        raise KeyError("Such key does not exist")
    
    def filter_by_healthsate(self, healthstate:HealthState):
        healthstate = EnumParser.parse(healthstate, HealthState).value
        return FeaturesDataset([dataset for dataset in self.participant_datasets if dataset.subject.health_state == healthstate])

    
    
    def to_long_dataframe(self):
        rows = []

        for participant_dataset in self.participant_datasets :
            subject_id = participant_dataset.subject.id
            subject_mmse = participant_dataset.subject.mmse
            subject_age = participant_dataset.subject.age
            subject_health = participant_dataset.subject.health_state
            features_df = participant_dataset.features_df

            df_long = (
                features_df
                .reset_index(names="channel")
                .melt(
                    id_vars="channel",
                    var_name="feature",
                    value_name="value"
                )
        )

            df_long["subject_id"] = subject_id
            df_long["subject_age"] = subject_age
            df_long["subject_health"] = subject_health
            df_long["subject_mmse"] = subject_mmse
        

            rows.append(df_long)

        big_df = pd.concat(rows, ignore_index=True)

        return big_df
    
    def to_wide_dataframe(self):
        rows = []

        for participant_dataset in self.participant_datasets:
            subject_id = participant_dataset.subject.id
            subject_health = participant_dataset.subject.health_state
            features_df = participant_dataset.features_df
            subject_mmse = participant_dataset.subject.mmse

            row = {
                "subject_id": subject_id,
                "subject_health": subject_health,
                "subject_mmse": subject_mmse
            }

            for channel in features_df.index:
                for feature in features_df.columns:
                    col_name = f"{channel}_{feature}"
                    row[col_name] = features_df.loc[channel, feature]

            rows.append(row)

        big_df = pd.DataFrame(rows)

        return big_df
    
    def to_subject_dataframe(self):
        rows = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject

            rows.append({
                "subject_id": subject.id,
                "subject_health": subject.health_state,
                "subject_age": subject.age,
                "subject_mmse": subject.mmse,
            })

        return pd.DataFrame(rows)
    
    @property
    def mean_feature_df(self):
        return DataframeHelpers.mean([dataset.features_df for dataset in self.participant_datasets])
    

    @property
    def selector(self):
        return SampleSelector(self)



