import numpy as np
from preprocessing.step.base import PreprocessingStep


class HampelFilterStep(PreprocessingStep):
    """
    Hampel filter avec fenêtre locale tronquée aux bords.
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
    def _hampel_1d(x: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if n == 0:
            return x.copy()

        half_window = window_size // 2
        y = x.copy()
        eps = np.finfo(float).eps

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

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy()
        raw.load_data()

        data = raw.get_data()
        filtered = np.empty_like(data, dtype=float)

        for ch_idx in range(data.shape[0]):
            filtered[ch_idx] = self._hampel_1d(
                data[ch_idx],
                window_size=self._window_size,
                n_sigma=self._n_sigma,
            )

        raw._data[:] = filtered
        new_eeg._raw = raw
        return new_eeg