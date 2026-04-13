from __future__ import annotations

import mne
import numpy as np
import scipy.ndimage

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class HampelFilterStep(PreprocessingStep):
    """
    Hampel filter rapide avec :
    - coeur du signal traité via scipy.ndimage.median_filter
    - bords corrigés exactement avec fenêtre tronquée
    """

    def __init__(self, window_size: int = 2001, n_sigma: float = 3.0):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if n_sigma <= 0:
            raise ValueError("n_sigma must be > 0")

        self._window_size = int(window_size)
        self._n_sigma = float(n_sigma)

    @property
    def name(self) -> str:
        return "hampel_filter"

    @property
    def params(self) -> dict:
        return {
            "window_size": self._window_size,
            "n_sigma": self._n_sigma,
        }

    @staticmethod
    def _fix_border_medians_truncated(
        x: np.ndarray,
        med: np.ndarray,
        half_window: int,
    ) -> None:
        n = x.size
        if n == 0:
            return

        left_stop = min(half_window, n)
        for i in range(left_stop):
            right = min(n, i + half_window + 1)
            med[i] = np.median(x[:right])

        right_start = max(left_stop, n - half_window)
        for i in range(right_start, n):
            left = max(0, i - half_window)
            med[i] = np.median(x[left:])

    @staticmethod
    def _fix_border_mads_truncated(
        x: np.ndarray,
        med: np.ndarray,
        mad: np.ndarray,
        half_window: int,
    ) -> None:
        n = x.size
        if n == 0:
            return

        left_stop = min(half_window, n)
        for i in range(left_stop):
            right = min(n, i + half_window + 1)
            window = x[:right]
            mad[i] = np.median(np.abs(window - med[i]))

        right_start = max(left_stop, n - half_window)
        for i in range(right_start, n):
            left = max(0, i - half_window)
            window = x[left:]
            mad[i] = np.median(np.abs(window - med[i]))

    @staticmethod
    def _hampel_1d(x: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = x.size

        if n == 0 or window_size == 1:
            return x.copy()

        half_window = window_size // 2
        eps = np.finfo(np.float64).eps

        if n <= window_size:
            y = x.copy()
            for i in range(n):
                left = max(0, i - half_window)
                right = min(n, i + half_window + 1)
                window = x[left:right]

                med = np.median(window)
                mad = np.median(np.abs(window - med))
                sigma = max(1.4826 * mad, eps)

                if abs(x[i] - med) > n_sigma * sigma:
                    y[i] = med
            return y

        med = scipy.ndimage.median_filter(
            x,
            size=window_size,
            mode="nearest",
        ).astype(np.float64, copy=False)

        HampelFilterStep._fix_border_medians_truncated(x, med, half_window)

        abs_dev = np.abs(x - med)

        mad = scipy.ndimage.median_filter(
            abs_dev,
            size=window_size,
            mode="nearest",
        ).astype(np.float64, copy=False)

        HampelFilterStep._fix_border_mads_truncated(x, med, mad, half_window)

        sigma = 1.4826 * mad
        np.maximum(sigma, eps, out=sigma)

        mask = np.abs(x - med) > (n_sigma * sigma)

        y = x.copy()
        y[mask] = med[mask]
        return y

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        if not raw.preload:
            raw.load_data(verbose=False)

        data = raw._data

        for ch_idx in range(data.shape[0]):
            data[ch_idx] = self._hampel_1d(
                data[ch_idx],
                window_size=self._window_size,
                n_sigma=self._n_sigma,
            )

        return raw