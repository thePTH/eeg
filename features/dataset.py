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
        vmin = np.min(values)
        vmax = np.max(values)

        fig, ax = plt.subplots(figsize=figsize)
        im, _ = mne.viz.plot_topomap(values, info, ch_type="eeg", show=False, sensors=True, axes=ax, contours=contours, cmap=cmap, vlim=(vmin, vmax))
        fig.colorbar(im, ax=ax)
        figure_title = title if title else feature_name
        subject = self.subject
        subject_description = f"Subject {subject.id} : {subject.health_state} | MMSE : {subject.mmse} | age : {subject.age}Y | gender : {subject.gender}"

        figure_subtitle = sub_title if sub_title else subject_description

        # titre principal centré (aligné avec la colorbar)
        fig.suptitle(
            figure_title,
            fontsize=16,
            y=0.98
        )

        # description en bas de figure
        fig.text(
            0.5, 0.02,
            figure_subtitle,
            ha="center",
            fontsize=10,
            color="gray"
        )

        plt.show()






class SingleParticipantProcessedFeatureDatasetactory:
    @staticmethod
    def build(extraction_result:FeatureExtractionResult):
        features_df = extraction_result.dataframe
        ppc_raw_data = extraction_result.ppc_result.dense_data_raw
        subject_dico = extraction_result.eeg.source.subject.to_dict()
        pipeline_name = extraction_result.eeg.pipeline_name
        eeg_info_dico = extraction_result.eeg.info.to_json_dict()
        return SingleParticipantProcessedFeatureDataset(features_df, ppc_raw_data, subject_dico, pipeline_name, eeg_info_dico)
    




class FeaturesDataset:
    def __init__(self, participant_datasets:list[SingleParticipantProcessedFeatureDataset]):
        self.participant_datasets = participant_datasets

    @property
    def subjects(self):
        return [dataset.subject for dataset in self.participant_datasets]
    
    @property
    def ch_names(self) -> list[str]:
        return self.participant_datasets[0].eeg_info_dico["ch_names"]
    
    @property
    def groups(self):
        return set([subject.group for subject in self.subjects])
    
    
    def to_long_dataframe(self):
        rows = []

        for participant_dataset in self.participant_datasets :
            subject_id = participant_dataset.subject.id
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
        df_long["subject_health"] = subject_health
        

        rows.append(df_long)

        big_df = pd.concat(rows, ignore_index=True)

        return big_df
    
    def to_wide_dataframe(self):
        rows = []

        for participant_dataset in self.participant_datasets:
            subject_id = participant_dataset.subject.id
            subject_health = participant_dataset.subject.health_state
            features_df = participant_dataset.features_df

            row = {
                "subject_id": subject_id,
                "subject_health": subject_health,
            }

            for channel in features_df.index:
                for feature in features_df.columns:
                    col_name = f"{channel}_{feature}"
                    row[col_name] = features_df.loc[channel, feature]

            rows.append(row)

        big_df = pd.DataFrame(rows)

        return big_df
