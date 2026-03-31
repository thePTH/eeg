import numpy as np
from preprocessing.step.base import PreprocessingStep


import numpy as np
import scipy.ndimage


class HampelFilterStep(PreprocessingStep):
    """
    Hampel filter rapide avec :
    - coeur du signal traité via scipy.ndimage.median_filter (très rapide)
    - bords corrigés exactement avec fenêtre tronquée

    Paramètres paper-consistent :
    - n_sigma = 3
    - Q = 1000 samples
    - longueur totale de fenêtre = 2*Q + 1 = 2001
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
    def _fix_border_medians_truncated(x: np.ndarray, med: np.ndarray, half_window: int) -> None:
        """
        Corrige les médianes aux bords pour reproduire exactement une fenêtre tronquée.
        Cette fonction modifie `med` en place.
        """
        n = x.size
        if n == 0:
            return

        # Bord gauche
        left_stop = min(half_window, n)
        for i in range(left_stop):
            right = min(n, i + half_window + 1)
            med[i] = np.median(x[:right])

        # Bord droit
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
        """
        Corrige les MAD aux bords pour reproduire exactement une fenêtre tronquée.
        Cette fonction modifie `mad` en place.
        """
        n = x.size
        if n == 0:
            return

        # Bord gauche
        left_stop = min(half_window, n)
        for i in range(left_stop):
            right = min(n, i + half_window + 1)
            window = x[:right]
            mad[i] = np.median(np.abs(window - med[i]))

        # Bord droit
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

        # Si le signal est très court, on fait le calcul exact directement
        # (ça évite des optimisations inutiles sur petits cas)
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

        # 1) Médiane glissante rapide sur tout le signal.
        # mode="nearest" est seulement une approximation de bord,
        # corrigée ensuite pour retrouver la fenêtre tronquée exacte.
        med = scipy.ndimage.median_filter(
            x,
            size=window_size,
            mode="nearest",
        ).astype(np.float64, copy=False)

        # Correction exacte des bords
        HampelFilterStep._fix_border_medians_truncated(x, med, half_window)

        # 2) MAD glissante rapide
        abs_dev = np.abs(x - med)

        mad = scipy.ndimage.median_filter(
            abs_dev,
            size=window_size,
            mode="nearest",
        ).astype(np.float64, copy=False)

        # Correction exacte des bords
        HampelFilterStep._fix_border_mads_truncated(x, med, mad, half_window)

        # 3) Test Hampel vectorisé
        sigma = 1.4826 * mad
        np.maximum(sigma, eps, out=sigma)

        mask = np.abs(x - med) > (n_sigma * sigma)

        y = x.copy()
        y[mask] = med[mask]
        return y

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy()
        

        # On travaille directement en place pour éviter une allocation 2D supplémentaire
        data = raw._data

        for ch_idx in range(data.shape[0]):
            data[ch_idx] = self._hampel_1d(
                data[ch_idx],
                window_size=self._window_size,
                n_sigma=self._n_sigma,
            )

        new_eeg._raw = raw
        return new_eeg