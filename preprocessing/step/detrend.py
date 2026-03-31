import numpy as np
import scipy.signal
from preprocessing.step.base import PreprocessingStep
from eeg.data import EEGData


class DetrendStep(PreprocessingStep):
    """
    Ancien detrend global conservé pour compatibilité.
    order=0 -> detrend constant
    order=1 -> detrend linéaire global
    """

    def __init__(self, order: int):
        self._order = order

    @property
    def name(self) -> str:
        return "detrend"

    @property
    def params(self) -> dict:
        return {"order": self._order}

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy()

        detrend_type = "constant" if self._order == 0 else "linear"
        raw.load_data()
        raw.apply_function(
            lambda signal: scipy.signal.detrend(signal, axis=-1, type=detrend_type)
        )

        new_eeg._raw = raw
        return new_eeg


class LocalDetrendStep(PreprocessingStep):
    """
    Local detrending inspiré de locdetrend (Chronux-like).
    On balaie le signal avec des fenêtres glissantes, on ajuste une droite
    localement, puis on moyenne les corrections sur les zones de recouvrement.

    Paramètres conseillés pour coller au papier :
    - window_sec = 0.06
    - step_sec   = 0.015
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
    def _local_detrend_1d(x: np.ndarray, fs: float, window_sec: float, step_sec: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if n == 0:
            return x.copy()

        window = max(3, int(round(window_sec * fs)))
        step = max(1, int(round(step_sec * fs)))

        if window > n:
            # Si le signal est trop court, on retombe proprement sur un detrend linéaire global.
            return scipy.signal.detrend(x, type="linear")

        corrected_sum = np.zeros(n, dtype=float)
        weights = np.zeros(n, dtype=float)

        starts = list(range(0, n - window + 1, step))
        if starts[-1] != n - window:
            starts.append(n - window)

        for start in starts:
            stop = start + window
            segment = x[start:stop]

            t = np.arange(window, dtype=float)
            # Ajustement affine local
            coeffs = np.polyfit(t, segment, deg=1)
            trend = np.polyval(coeffs, t)
            corrected = segment - trend

            corrected_sum[start:stop] += corrected
            weights[start:stop] += 1.0

        # Sécurité numérique
        valid = weights > 0
        out = x.copy()
        out[valid] = corrected_sum[valid] / weights[valid]

        # En principe, tout est couvert, mais on protège les cas limites.
        if np.any(~valid):
            out[~valid] = scipy.signal.detrend(x[~valid], type="linear")

        return out

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy()
        raw.load_data()

        fs = float(eeg_data.sampling_frequency)

        raw.apply_function(
            lambda signal: self._local_detrend_1d(
                signal,
                fs=fs,
                window_sec=self._window_sec,
                step_sec=self._step_sec,
            ),
            channel_wise=True,
        )

        new_eeg._raw = raw
        return new_eeg