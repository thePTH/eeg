from features.factory import FeatureExtractionResult
import mne
import matplotlib.pyplot as plt
import numpy as np

class ExtractedFeatureHeatmapFactory:
    def __init__(self, extraction_result:FeatureExtractionResult):
        self.extraction_result = extraction_result


    def plot(self, feature_name:str, title:str=None, sub_title:str=None, figsize=(7,6), contours=7, cmap="RdBu_r"):
        info = self.extraction_result.eeg.info
        values = self.extraction_result.values(feature_name)
        vmin = np.min(values)
        vmax = np.max(values)


        fig, ax = plt.subplots(figsize=figsize)
        im, _ = mne.viz.plot_topomap(values, info, ch_type="eeg", show=False, sensors=True, axes=ax, contours=contours, cmap=cmap, vlim=(vmin, vmax))
        fig.colorbar(im, ax=ax)
        figure_title = title if title else feature_name
        subject = self.extraction_result.eeg.source.subject

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


