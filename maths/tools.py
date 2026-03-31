from __future__ import annotations

from constants import EPS
import numpy as np
from typing import Iterable



class SignalTools:
    """Outils de base pour manipuler et valider un signal 1D."""

    @staticmethod
    def as_1d_float_array(x: Iterable[float]) -> np.ndarray:
        """
        Convertit une entrée quelconque en tableau numpy 1D de floats.

        Paramètres
        ----------
        x :
            Iterable contenant les échantillons du signal.

        Retour
        ------
        np.ndarray
            Tableau 1D de dtype float64.

        Lève
        ----
        ValueError
            Si le signal est vide ou contient des NaN / inf.
        """
        arr = np.asarray(list(x), dtype=np.float64).ravel()

        if arr.size == 0:
            raise ValueError("Le signal est vide.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Le signal contient des NaN ou des inf.")

        return arr

    @staticmethod
    def rms(x: np.ndarray) -> float:
        """
        Root Mean Square.
        """
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def normalized_binary_sequence(x: np.ndarray) -> np.ndarray:
        """
        Binarisation simple pour LZC :
        1 si x >= médiane, 0 sinon.
        """
        med = np.median(x)
        return (x >= med).astype(np.uint8)


class EmbeddingTools:
    """Outils liés à la reconstruction d'espace d'état."""

    @staticmethod
    def sliding_embed(x: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
        """
        Reconstruction d'espace d'état.

        Retourne une matrice de shape (n_vectors, m), où chaque ligne
        correspond à un vecteur retardé.

        Paramètres
        ----------
        x :
            Signal 1D.
        m :
            Dimension d'embedding.
        tau :
            Retard.

        Retour
        ------
        np.ndarray
            Matrice d'embedding.

        Lève
        ----
        ValueError
            Si les paramètres sont invalides ou si le signal est trop court.
        """
        if m < 1 or tau < 1:
            raise ValueError("m et tau doivent être >= 1.")

        n = len(x) - (m - 1) * tau
        if n <= 1:
            raise ValueError("Signal trop court pour la reconstruction d'état.")

        return np.column_stack([x[i:i + n] for i in range(0, m * tau, tau)])