from __future__ import annotations

import mne
import numpy as np
import scipy.signal

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class DetrendStep(PreprocessingStep):
    """
    Detrend global.
    - order=0 : suppression de la composante constante
    - order=1 : detrend linéaire global
    """

    def __init__(self, order: int):
        if order not in (0, 1):
            raise ValueError("order must be 0 or 1")
        self._order = order

    @property
    def name(self) -> str:
        return "detrend"

    @property
    def params(self) -> dict:
        return {"order": self._order}

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        detrend_type = "constant" if self._order == 0 else "linear"

        raw.apply_function(
            lambda signal: scipy.signal.detrend(
                signal,
                axis=-1,
                type=detrend_type,
            ),
            channel_wise=True,
        )

        return raw


class LocalDetrendStep(PreprocessingStep):
    """
    Local detrending inspiré de locdetrend.

    L'idée est de retirer une tendance affine locale estimée sur des fenêtres
    glissantes qui se recouvrent.
    """

    def __init__(self, window_sec: float = 0.06, step_sec: float = 0.015):
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        if step_sec <= 0:
            raise ValueError("step_sec must be > 0")
        if step_sec > window_sec:
            raise ValueError("step_sec must be <= window_sec")

        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)

    @property
    def name(self) -> str:
        return "local_detrend"

    @property
    def params(self) -> dict:
        return {
            "window_sec": self._window_sec,
            "step_sec": self._step_sec,
        }

    @staticmethod
    def _local_detrend_1d(
        x: np.ndarray,
        fs: float,
        window_sec: float,
        step_sec: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = x.size

        if n == 0:
            return x.copy()

        window = max(3, int(round(window_sec * fs)))
        step = max(1, int(round(step_sec * fs)))

        if window > n:
            return scipy.signal.detrend(x, type="linear")

        starts = np.arange(0, n - window + 1, step, dtype=np.int64)
        if starts[-1] != n - window:
            starts = np.append(starts, n - window)

        W = float(window)
        t_sum = W * (W - 1.0) / 2.0
        t2_sum = (W - 1.0) * W * (2.0 * W - 1.0) / 6.0
        denom = W * t2_sum - t_sum * t_sum

        idx = np.arange(n, dtype=np.float64)

        csum_x = np.empty(n + 1, dtype=np.float64)
        csum_x[0] = 0.0
        csum_x[1:] = np.cumsum(x)

        csum_gx = np.empty(n + 1, dtype=np.float64)
        csum_gx[0] = 0.0
        csum_gx[1:] = np.cumsum(idx * x)

        stops = starts + window

        sum_y = csum_x[stops] - csum_x[starts]
        sum_gx = csum_gx[stops] - csum_gx[starts]
        sum_ty = sum_gx - starts * sum_y

        slopes = (W * sum_ty - t_sum * sum_y) / denom
        intercepts_local = (sum_y / W) - slopes * (t_sum / W)

        c_global = intercepts_local - slopes * starts

        diff_w = np.zeros(n + 1, dtype=np.float64)
        diff_a = np.zeros(n + 1, dtype=np.float64)
        diff_c = np.zeros(n + 1, dtype=np.float64)

        np.add.at(diff_w, starts, 1.0)
        np.add.at(diff_w, stops, -1.0)

        np.add.at(diff_a, starts, slopes)
        np.add.at(diff_a, stops, -slopes)

        np.add.at(diff_c, starts, c_global)
        np.add.at(diff_c, stops, -c_global)

        weights = np.cumsum(diff_w[:-1])
        sum_a_cover = np.cumsum(diff_a[:-1])
        sum_c_cover = np.cumsum(diff_c[:-1])

        out = x.copy()
        valid = weights > 0

        trend_avg = np.zeros(n, dtype=np.float64)
        trend_avg[valid] = (
            idx[valid] * sum_a_cover[valid] + sum_c_cover[valid]
        ) / weights[valid]

        out[valid] = x[valid] - trend_avg[valid]

        if np.any(~valid):
            out[~valid] = scipy.signal.detrend(x[~valid], type="linear")

        return out

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        fs = float(raw.info["sfreq"])

        raw.apply_function(
            lambda signal: self._local_detrend_1d(
                signal,
                fs=fs,
                window_sec=self._window_sec,
                step_sec=self._step_sec,
            ),
            channel_wise=True,
        )

        return raw