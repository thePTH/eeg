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
       
        raw.apply_function(
            lambda signal: scipy.signal.detrend(signal, axis=-1, type=detrend_type)
        )

        new_eeg._raw = raw
        return new_eeg


import numpy as np
import scipy.signal


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
    def _local_detrend_1d(
        x: np.ndarray,
        fs: float,
        window_sec: float,
        step_sec: float,
    ) -> np.ndarray:
        """
        Version optimisée :
        - pas de np.polyfit dans une boucle
        - calcul vectorisé des régressions locales
        - agrégation des contributions par tableaux de différences

        Complexité bien plus faible que l'approche segment par segment.
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size

        if n == 0:
            return x.copy()

        window = max(3, int(round(window_sec * fs)))
        step = max(1, int(round(step_sec * fs)))

        if window > n:
            # Repli propre si le signal est trop court
            return scipy.signal.detrend(x, type="linear")

        # ------------------------------------------------------------------
        # 1) Construction des positions de départ des fenêtres
        # ------------------------------------------------------------------
        starts = np.arange(0, n - window + 1, step, dtype=np.int64)
        if starts[-1] != n - window:
            starts = np.append(starts, n - window)

        # ------------------------------------------------------------------
        # 2) Pré-calculs pour la régression linéaire locale sur t = 0..window-1
        #
        #    slope = (W * sum(t*y) - sum(t) * sum(y)) / (W * sum(t^2) - sum(t)^2)
        #    intercept_local = mean(y) - slope * mean(t)
        #
        #    Pour chaque fenêtre [s, s+W):
        #    sum(t*y) = sum(g*x[g]) - s * sum(x[g]), où g est l'indice global.
        # ------------------------------------------------------------------
        W = float(window)
        t_sum = W * (W - 1.0) / 2.0
        t2_sum = (W - 1.0) * W * (2.0 * W - 1.0) / 6.0
        denom = W * t2_sum - t_sum * t_sum

        # Sommes cumulées de x et de g*x[g]
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

        # ------------------------------------------------------------------
        # 3) Passage de la droite locale a*(i-start) + b
        #    à une forme globale :
        #
        #       a*i + c, avec c = b - a*start
        #
        #    Ainsi, pour un échantillon i, la moyenne des tendances prédites
        #    sur les fenêtres qui le couvrent vaut :
        #
        #       (i * sum(a) + sum(c)) / weight
        # ------------------------------------------------------------------
        c_global = intercepts_local - slopes * starts

        # ------------------------------------------------------------------
        # 4) Sommes des contributions sur intervalles via tableaux de différences
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 5) Reconstruction finale
        # ------------------------------------------------------------------
        out = x.copy()
        valid = weights > 0

        trend_avg = np.zeros(n, dtype=np.float64)
        trend_avg[valid] = (
            idx[valid] * sum_a_cover[valid] + sum_c_cover[valid]
        ) / weights[valid]

        out[valid] = x[valid] - trend_avg[valid]

        # Sécurité sur cas limites
        if np.any(~valid):
            out[~valid] = scipy.signal.detrend(x[~valid], type="linear")

        return out

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy()
        

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