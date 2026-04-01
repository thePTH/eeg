from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy
import matplotlib.pyplot as plt


class SpectralTools:
    """Outils spectraux de bas niveau."""

    @staticmethod
    def bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi)

        if not np.any(mask):
            return 0.0

        return float(np.trapezoid(psd[mask], freqs[mask]))


class PSDAnalysisResult:
    def __init__(self, psd: np.ndarray, freqs: np.ndarray):
        self.psd = psd
        self.freqs = freqs

    def plot(
        self,
        db: bool = True,
        log_freq: bool = False,
        title: str = "Power Spectral Density",
        show: bool = True,
    ):
        psd = self.psd.copy()

        if db:
            psd = 10 * np.log10(psd + 1e-20)

        plt.figure(figsize=(8, 4))

        if log_freq:
            plt.semilogx(self.freqs, psd)
        else:
            plt.plot(self.freqs, psd)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB/Hz)" if db else "PSD")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        if show:
            plt.show()


    def to_dict(self) -> dict[float, float]:
        return {freq : power for freq, power in zip(self.freqs, self.psd)}


@dataclass(slots=True, frozen=True)
class SignalSpectralAnalysisResult:
    freqs: np.ndarray
    psd: np.ndarray
    fft_freqs: np.ndarray
    fft_magnitude: np.ndarray

    band_powers: dict[str, float]
    relative_band_powers: dict[str, float]
    total_power: float

    delta_power: float
    theta_power: float
    alpha_power: float
    beta_power: float
    gamma_power: float

    theta_beta_ratio: float
    theta_alpha_ratio: float
    gamma_alpha_ratio: float
    spectral_power_ratio: float

    dominant_frequency_full: float
    dominant_frequency_delta: float
    dominant_frequency_theta: float
    dominant_frequency_alpha: float
    dominant_frequency_beta: float
    dominant_frequency_gamma: float

    centroid: float
    spread: float
    skewness: float
    kurtosis: float
    rolloff_95: float
    flux: float

    @property
    def psd_analysis_result(self):
        return PSDAnalysisResult(self.psd, self.freqs)


@dataclass(slots=True, frozen=True)
class SignalSpectralAnalysisParameters:
    bands: dict[str, tuple[float, float]]
    spectral_flux_segment_sec: float
    psd_time_halfbandwidth_product: float


class SignalSpectralAnalysisEngine:
    """
    Moteur de calcul de l'analyse spectrale.

    Philosophie de cette version :
    - PSD multi-taper conservée pour les mesures PSD/globales
    - voie séparée dédiée aux power ratios pour se rapprocher davantage
      de la logique du papier tout en gardant TES bandes fréquentielles
    """

    def __init__(self, x: np.ndarray, fs: float, params: SignalSpectralAnalysisParameters):
        self.x = np.asarray(x, dtype=float)
        self.fs = float(fs)
        self.params = params

        if self.fs <= 0:
            raise ValueError("La fréquence d'échantillonnage doit être > 0.")
        if self.x.ndim != 1:
            raise ValueError("Le signal doit être 1D.")

    def _compute_psd_multitaper(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.x)
        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        Kmax = 1
        psd_time_halfbandwidth_product = self.params.psd_time_halfbandwidth_product

        tapers = scipy.signal.windows.dpss(
            M=n,
            NW=psd_time_halfbandwidth_product,
            Kmax=Kmax,
            sym=False,
            norm=2,
        )

        taper = np.asarray(tapers[0], dtype=float)

        x_tapered = self.x * taper
        X = np.fft.rfft(x_tapered)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        taper_energy = np.sum(taper ** 2)
        if taper_energy <= 0.0:
            return freqs, np.zeros_like(freqs, dtype=float)

        psd = (np.abs(X) ** 2) / (self.fs * taper_energy)

        if n > 1:
            if n % 2 == 0:
                if psd.size > 2:
                    psd[1:-1] *= 2.0
            else:
                if psd.size > 1:
                    psd[1:] *= 2.0

        return freqs, psd

    def _compute_fft(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.x)
        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        mag = np.abs(np.fft.rfft(self.x))
        return freqs, mag

    def _compute_band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
        return {
            name: SpectralTools.bandpower_from_psd(freqs, psd, band)
            for name, band in self.params.bands.items()
        }

    def _compute_total_power(self, freqs: np.ndarray, psd: np.ndarray, band_powers: dict[str, float]) -> float:
        if "full" in band_powers:
            return float(band_powers["full"])

        if freqs.size == 0 or psd.size == 0:
            return 0.0

        return float(np.trapezoid(psd, freqs))

    def _compute_relative_band_powers(
        self,
        band_powers: dict[str, float],
        total_power: float,
    ) -> dict[str, float]:
        if total_power <= 0.0:
            return {name: 0.0 for name in band_powers}

        return {
            name: float(power / total_power)
            for name, power in band_powers.items()
        }

    def _get_band_power(self, band_powers: dict[str, float], band_name: str) -> float:
        return float(band_powers.get(band_name, 0.0))

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if denominator == 0.0:
            return 0.0
        return float(numerator / denominator)

    def _next_pow2(self, n: int) -> int:
        if n <= 1:
            return 1
        return 2 ** int(np.ceil(np.log2(n)))

    def _compute_ratio_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Voie spectrale dédiée aux power ratios :
        - retrait de moyenne
        - fenêtre de Hamming
        - zero-padding à la prochaine puissance de 2
        - rFFT
        - puissance one-sided |X(f)|^2
        """
        x = np.asarray(self.x, dtype=float)
        n = len(x)

        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        x = x - np.mean(x)

        nfft = self._next_pow2(n)
        window = np.hamming(n)
        xw = x * window

        X = np.fft.rfft(xw, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
        power = np.abs(X) ** 2

        return freqs, power

    def _band_power_from_spectrum(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        band: tuple[float, float],
    ) -> float:
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi)

        if not np.any(mask):
            return 0.0

        return float(np.sum(power[mask]))

    def _compute_dominant_frequency(
        self,
        fft_freqs: np.ndarray,
        fft_magnitude: np.ndarray,
        low: float,
        high: float,
    ) -> float:
        if fft_freqs.size == 0 or fft_magnitude.size == 0:
            return 0.0

        mask = (fft_freqs >= low) & (fft_freqs <= high)
        if not np.any(mask):
            return 0.0

        sub_f = fft_freqs[mask]
        sub_m = fft_magnitude[mask]
        return float(sub_f[np.argmax(sub_m)])

    def _compute_centroid(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        if freqs.size == 0 or psd.size == 0:
            return 0.0

        denom = np.sum(psd)
        if denom <= 0.0:
            return 0.0

        return float(np.sum(freqs * psd) / denom)

    def _compute_spread(self, freqs: np.ndarray, psd: np.ndarray, centroid: float) -> float:
        if freqs.size == 0 or psd.size == 0:
            return 0.0

        denom = np.sum(psd)
        if denom <= 0.0:
            return 0.0

        var = np.sum(psd * (freqs - centroid) ** 2) / denom
        return float(np.sqrt(max(var, 0.0)))

    def _compute_skewness(self, freqs: np.ndarray, psd: np.ndarray, centroid: float, spread: float) -> float:
        if freqs.size == 0 or psd.size == 0:
            return 0.0

        denom = np.sum(psd)
        if denom <= 0.0 or spread <= 0.0:
            return 0.0

        m3 = np.sum(psd * (freqs - centroid) ** 3) / denom
        return float(m3 / (spread ** 3))

    def _compute_kurtosis(self, freqs: np.ndarray, psd: np.ndarray, centroid: float, spread: float) -> float:
        if freqs.size == 0 or psd.size == 0:
            return 0.0

        denom = np.sum(psd)
        if denom <= 0.0 or spread <= 0.0:
            return 0.0

        m4 = np.sum(psd * (freqs - centroid) ** 4) / denom
        return float(m4 / (spread ** 4))

    def _compute_rolloff_95(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        if freqs.size == 0 or psd.size == 0:
            return 0.0

        cumulative = np.cumsum(psd)
        if cumulative.size == 0 or cumulative[-1] <= 0:
            return 0.0

        threshold = 0.95 * cumulative[-1]
        idx = np.searchsorted(cumulative, threshold)
        idx = min(idx, len(freqs) - 1)
        return float(freqs[idx])

    def _compute_flux(self) -> float:
        n = len(self.x)
        if n < 2:
            return 0.0

        seg_len = max(8, int(round(self.params.spectral_flux_segment_sec * self.fs)))
        nseg = n // seg_len

        if nseg < 2:
            return 0.0

        segments = self.x[: nseg * seg_len].reshape(nseg, seg_len)
        mags = np.abs(np.fft.rfft(segments, axis=1))
        diffs = np.diff(mags, axis=0)
        flux = np.mean(np.sum(diffs ** 2, axis=1))
        return float(flux)

    def compute(self) -> SignalSpectralAnalysisResult:
        freqs, psd = self._compute_psd_multitaper()
        fft_freqs, fft_magnitude = self._compute_fft()

        # Sorties PSD classiques conservées
        band_powers = self._compute_band_powers(freqs, psd)
        total_power = self._compute_total_power(freqs, psd, band_powers)
        relative_band_powers = self._compute_relative_band_powers(band_powers, total_power)

        delta_power = self._get_band_power(band_powers, "delta")
        theta_power = self._get_band_power(band_powers, "theta")
        alpha_power = self._get_band_power(band_powers, "alpha")
        beta_power = self._get_band_power(band_powers, "beta")
        gamma_power = self._get_band_power(band_powers, "gamma")

        # Nouvelle voie dédiée aux ratios
        ratio_freqs, ratio_power = self._compute_ratio_spectrum()

        delta_ratio_power = self._band_power_from_spectrum(
            ratio_freqs, ratio_power, self.params.bands["delta"]
        )
        theta_ratio_power = self._band_power_from_spectrum(
            ratio_freqs, ratio_power, self.params.bands["theta"]
        )
        alpha_ratio_power = self._band_power_from_spectrum(
            ratio_freqs, ratio_power, self.params.bands["alpha"]
        )
        beta_ratio_power = self._band_power_from_spectrum(
            ratio_freqs, ratio_power, self.params.bands["beta"]
        )
        gamma_ratio_power = self._band_power_from_spectrum(
            ratio_freqs, ratio_power, self.params.bands["gamma"]
        )

        theta_beta_ratio = self._safe_ratio(theta_ratio_power, beta_ratio_power)
        theta_alpha_ratio = self._safe_ratio(theta_ratio_power, alpha_ratio_power)
        gamma_alpha_ratio = self._safe_ratio(gamma_ratio_power, alpha_ratio_power)

        # Version "paper-consistent light" :
        # on enlève gamma du dénominateur et on oppose activité plus rapide vs plus lente
        spectral_power_ratio = self._safe_ratio(
            alpha_ratio_power + beta_ratio_power,
            theta_ratio_power + delta_ratio_power,
        )

        full_band = self.params.bands.get("full", (0.0, self.fs / 2.0))
        delta_band = self.params.bands.get("delta", (0.0, 0.0))
        theta_band = self.params.bands.get("theta", (0.0, 0.0))
        alpha_band = self.params.bands.get("alpha", (0.0, 0.0))
        beta_band = self.params.bands.get("beta", (0.0, 0.0))
        gamma_band = self.params.bands.get("gamma", (0.0, 0.0))

        dominant_frequency_full = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, full_band[0], full_band[1]
        )
        dominant_frequency_delta = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, delta_band[0], delta_band[1]
        )
        dominant_frequency_theta = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, theta_band[0], theta_band[1]
        )
        dominant_frequency_alpha = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, alpha_band[0], alpha_band[1]
        )
        dominant_frequency_beta = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, beta_band[0], beta_band[1]
        )
        dominant_frequency_gamma = self._compute_dominant_frequency(
            fft_freqs, fft_magnitude, gamma_band[0], gamma_band[1]
        )

        centroid = self._compute_centroid(freqs, psd)
        spread = self._compute_spread(freqs, psd, centroid)
        skewness = self._compute_skewness(freqs, psd, centroid, spread)
        kurtosis = self._compute_kurtosis(freqs, psd, centroid, spread)
        rolloff_95 = self._compute_rolloff_95(freqs, psd)
        flux = self._compute_flux()

        return SignalSpectralAnalysisResult(
            freqs=freqs,
            psd=psd,
            fft_freqs=fft_freqs,
            fft_magnitude=fft_magnitude,
            band_powers=band_powers,
            relative_band_powers=relative_band_powers,
            total_power=total_power,
            delta_power=delta_power,
            theta_power=theta_power,
            alpha_power=alpha_power,
            beta_power=beta_power,
            gamma_power=gamma_power,
            theta_beta_ratio=theta_beta_ratio,
            theta_alpha_ratio=theta_alpha_ratio,
            gamma_alpha_ratio=gamma_alpha_ratio,
            spectral_power_ratio=spectral_power_ratio,
            dominant_frequency_full=dominant_frequency_full,
            dominant_frequency_delta=dominant_frequency_delta,
            dominant_frequency_theta=dominant_frequency_theta,
            dominant_frequency_alpha=dominant_frequency_alpha,
            dominant_frequency_beta=dominant_frequency_beta,
            dominant_frequency_gamma=dominant_frequency_gamma,
            centroid=centroid,
            spread=spread,
            skewness=skewness,
            kurtosis=kurtosis,
            rolloff_95=rolloff_95,
            flux=flux,
        )