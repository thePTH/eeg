import pandas as pd
import numpy as np


class DataframeHelpers:
    """
    Classe utilitaire statique pour manipuler des DataFrames
    canal × features (typiquement EEG topographiques).

    Toutes les méthodes sont statiques.
    """

    # ================================
    # Validation structure DataFrames
    # ================================

    @staticmethod
    def validate_same_structure(df_list):
        """
        Vérifie que tous les DataFrames ont les mêmes index et colonnes.
        """

        if not df_list:
            raise ValueError("La liste de DataFrames est vide.")

        ref_index = df_list[0].index
        ref_columns = df_list[0].columns

        for i, df in enumerate(df_list[1:], start=1):

            if not df.index.equals(ref_index):
                raise ValueError(
                    f"Incohérence des index entre df[0] et df[{i}]"
                )

            if not df.columns.equals(ref_columns):
                raise ValueError(
                    f"Incohérence des colonnes entre df[0] et df[{i}]"
                )

    # ================================
    # Moyenne inter-DataFrames
    # ================================

    @staticmethod
    def mean(df_list):
        """
        Calcule la moyenne élément par élément d'une liste de DataFrames.

        Parameters
        ----------
        df_list : list[pd.DataFrame]

        Returns
        -------
        pd.DataFrame
        """

        DataframeHelpers.validate_same_structure(df_list)

        stacked = np.stack([df.values for df in df_list], axis=0)

        mean_array = np.nanmean(stacked, axis=0)

        return pd.DataFrame(
            mean_array,
            index=df_list[0].index,
            columns=df_list[0].columns,
        )

    # ================================
    # Médiane inter-DataFrames
    # ================================

    @staticmethod
    def median(df_list):
        """
        Médiane élément par élément.
        """

        DataframeHelpers.validate_same_structure(df_list)

        stacked = np.stack([df.values for df in df_list], axis=0)

        median_array = np.nanmedian(stacked, axis=0)

        return pd.DataFrame(
            median_array,
            index=df_list[0].index,
            columns=df_list[0].columns,
        )

    # ================================
    # Écart-type inter-DataFrames
    # ================================

    @staticmethod
    def std(df_list):
        """
        Écart-type élément par élément.
        """

        DataframeHelpers.validate_same_structure(df_list)

        stacked = np.stack([df.values for df in df_list], axis=0)

        std_array = np.nanstd(stacked, axis=0)

        return pd.DataFrame(
            std_array,
            index=df_list[0].index,
            columns=df_list[0].columns,
        )

    # ================================
    # Normalisation z-score
    # ================================

    @staticmethod
    def zscore(df):
        """
        Normalisation z-score colonne par colonne.
        """

        return (df - df.mean()) / df.std()

    # ================================
    # Min-max scaling
    # ================================

    @staticmethod
    def minmax(df):
        """
        Normalisation min-max colonne par colonne.
        """

        return (df - df.min()) / (df.max() - df.min())

    # ================================
    # Alignement sur un ordre de canaux
    # ================================

    @staticmethod
    def reorder_channels(df, channel_order):
        """
        Réordonne les lignes selon un ordre de canaux donné.

        Parameters
        ----------
        df : pd.DataFrame
        channel_order : list[str]

        Returns
        -------
        pd.DataFrame
        """

        missing = set(channel_order) - set(df.index)

        if missing:
            raise ValueError(
                f"Canaux manquants dans le DataFrame : {missing}"
            )

        return df.loc[channel_order]