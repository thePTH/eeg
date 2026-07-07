from __future__ import annotations

from typing import Iterable

import numpy as np

from constants import EPS


class SignalTools:
    """Utility methods for validating and manipulating one-dimensional signals."""

    @staticmethod
    def as_1d_float_array(x: Iterable[float]) -> np.ndarray:
        """
        Convert an iterable into a one-dimensional NumPy array of floats.

        Parameters
        ----------
        x : Iterable[float]
            Signal samples.

        Returns
        -------
        np.ndarray
            One-dimensional array with dtype ``float64``.

        Raises
        ------
        ValueError
            If the signal is empty or contains NaN or infinite values.
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
        Compute the Root Mean Square (RMS) of a signal.

        Parameters
        ----------
        x : np.ndarray
            Input signal.

        Returns
        -------
        float
            RMS value.
        """
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def normalized_binary_sequence(x: np.ndarray) -> np.ndarray:
        """
        Convert a signal into a binary sequence using its median.

        Samples greater than or equal to the median are mapped to 1, while
        samples below the median are mapped to 0.

        Parameters
        ----------
        x : np.ndarray
            Input signal.

        Returns
        -------
        np.ndarray
            Binary sequence with dtype ``uint8``.
        """
        med = np.median(x)

        return (x >= med).astype(np.uint8)


class EmbeddingTools:
    """Utility methods related to state-space reconstruction."""

    @staticmethod
    def sliding_embed(x: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
        """
        Reconstruct the state space using delay embedding.

        Each row of the returned matrix corresponds to a delayed embedding
        vector.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional input signal.
        m : int
            Embedding dimension.
        tau : int, default=1
            Time delay between embedding dimensions.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape ``(n_vectors, m)``.

        Raises
        ------
        ValueError
            If the embedding parameters are invalid or the signal is too short.
        """
        if m < 1 or tau < 1:
            raise ValueError("m et tau doivent être >= 1.")

        n = len(x) - (m - 1) * tau

        if n <= 1:
            raise ValueError("Signal trop court pour la reconstruction d'état.")

        return np.column_stack(
            [
                x[i : i + n]
                for i in range(0, m * tau, tau)
            ]
        )