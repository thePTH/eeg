from features.factory import FeatureExtractionResult
import mne
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import mne


class TopomapFactory:
    """
    Fabrique utilitaire pour afficher des topomaps EEG à partir :
    - d'un vecteur de valeurs par canal
    - d'un objet MNE Info (ou d'un dict JSON pour le reconstruire)

    Cette classe ne stocke pas d'état métier : elle sert uniquement
    à centraliser la logique de visualisation.
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
        sensors: bool = True
    ):
        """
        Affiche une topomap EEG.

        Parameters
        ----------
        values : array-like
            Tableau 1D de taille (n_channels,) contenant une valeur par canal.
        eeg_info : mne.Info or dict
            Objet MNE Info, ou dictionnaire JSON-compatible permettant
            de le reconstruire avec `mne.Info.from_json_dict`.
        title : str, optional
            Titre principal de la figure.
        sub_title : str, optional
            Sous-titre affiché en bas de la figure.
        figsize : tuple, default=(7, 6)
            Taille de la figure.
        contours : int, default=7
            Nombre de lignes de contour.
        cmap : str, default="RdBu_r"
            Colormap utilisée.
        vlim : tuple, optional
            Bornes (vmin, vmax). Si None, elles sont calculées automatiquement.
        sensors : bool, default=True
            Si True, affiche les positions des électrodes.
        show : bool, default=True
            Si True, appelle plt.show().

        Returns
        -------
        fig, ax
            Figure et axe matplotlib.
        """
        # Reconstruction éventuelle de l'objet Info
        if isinstance(eeg_info, dict):
            info = mne.Info.from_json_dict(eeg_info)
        else:
            info = eeg_info

        # Conversion robuste en numpy array 1D
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

        # Bornes de couleur
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






