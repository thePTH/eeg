from features.factory import FeatureExtractionResult

import matplotlib.pyplot as plt
import mne
import numpy as np


class TopomapFactory:
    """
    Utility factory for plotting EEG topomaps from:
    - a vector of channel-wise values;
    - an MNE Info object, or a JSON-compatible dictionary used to reconstruct it.

    This class does not store domain state. It only centralizes visualization
    logic.
    """

    @staticmethod
    def plot(
        values,
        eeg_info,
        title: str = None,
        sub_title: str = None,
        figsize=(7, 6),
        contours: int = 7,
        cmap: str = "RdBu_r",
        vlim: tuple = None,
        sensors: bool = True,
    ):
        """
        Plot an EEG topomap.

        Parameters
        ----------
        values : array-like
            One-dimensional array of shape ``(n_channels,)`` containing one
            value per EEG channel.
        eeg_info : mne.Info or dict
            MNE Info object, or JSON-compatible dictionary that can be
            reconstructed with ``mne.Info.from_json_dict``.
        title : str, optional
            Main figure title.
        sub_title : str, optional
            Subtitle displayed at the bottom of the figure.
        figsize : tuple, default=(7, 6)
            Figure size.
        contours : int, default=7
            Number of contour lines.
        cmap : str, default="RdBu_r"
            Colormap used for the topomap.
        vlim : tuple, optional
            Color limits ``(vmin, vmax)``. If None, limits are computed
            automatically.
        sensors : bool, default=True
            Whether to display electrode positions.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        if isinstance(eeg_info, dict):
            info = mne.Info.from_json_dict(eeg_info)
        else:
            info = eeg_info

        values = np.asarray(values)

        if values.ndim != 1:
            raise ValueError(
                f"`values` doit être un vecteur 1D de taille (n_channels,), "
                f"mais a la forme {values.shape}."
            )

        n_channels = len(info["ch_names"])

        if len(values) != n_channels:
            raise ValueError(
                f"Incohérence entre `values` et `eeg_info` : "
                f"{len(values)} valeurs fournies pour {n_channels} canaux EEG."
            )

        if vlim is None:
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            vlim = (vmin, vmax)

        fig, ax = plt.subplots(figsize=figsize)

        im, _ = mne.viz.plot_topomap(
            values,
            info,
            ch_type="eeg",
            show=False,
            sensors=sensors,
            axes=ax,
            contours=contours,
            cmap=cmap,
            vlim=vlim,
        )

        fig.colorbar(im, ax=ax)

        fig.suptitle(
            title if title is not None else "EEG Topomap",
            fontsize=16,
            y=0.98,
        )

        if sub_title is not None:
            fig.text(
                0.5,
                0.02,
                sub_title,
                ha="center",
                fontsize=10,
                color="gray",
            )

        plt.show()