from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


@dataclass(slots=True, frozen=True)
class SignalWaveletAnalysisResult:
    """
    Container exposing wavelet analysis results.

    This class does not perform any computation. It only stores raw
    coefficients and metrics already computed by `SignalWaveletAnalysisEngine`.
    """

    coeffs: list[np.ndarray]
    packet_leaves: list[np.ndarray]

    approximate_energy: float
    detail_energy: float
    relative_wavelet_energy: float

    packet_approximate_energy: float
    packet_detail_energy: float
    relative_wavelet_packet_energy: float


@dataclass(slots=True, frozen=True)
class SignalWaveletAnalysisParameters:
    """Parameters used by the wavelet analysis engine."""

    wavelet: str
    wavelet_level: int


class SignalWaveletAnalysisEngine:
    """Engine used to compute wavelet and wavelet packet decompositions."""

    def __init__(self, x: np.ndarray, params: SignalWaveletAnalysisParameters):
        self.x = np.asarray(x, dtype=float)
        self.params = params

    def _resolve_level(self) -> int:
        """Determine a valid and robust decomposition level."""
        wavelet = pywt.Wavelet(self.params.wavelet)
        max_level = pywt.dwt_max_level(len(self.x), wavelet.dec_len)

        if max_level < 1:
            return 1

        return min(self.params.wavelet_level, max_level)

    @staticmethod
    def _compute_approximate_energy(coeffs: list[np.ndarray]) -> float:
        """Compute the energy of the main approximation coefficient."""
        if len(coeffs) == 0:
            return 0.0

        return float(np.sum(np.square(coeffs[0])))

    @staticmethod
    def _compute_detail_energy(coeffs: list[np.ndarray]) -> float:
        """Compute the total energy of all detail coefficients."""
        if len(coeffs) <= 1:
            return 0.0

        return float(sum(np.sum(np.square(cd)) for cd in coeffs[1:]))

    @staticmethod
    def _compute_relative_wavelet_energy(coeffs: list[np.ndarray]) -> float:
        """Compute the maximum relative energy among wavelet coefficients."""
        if len(coeffs) == 0:
            return 0.0

        energies = np.asarray(
            [
                np.sum(np.square(c))
                for c in coeffs
            ],
            dtype=float,
        )
        total_energy = np.sum(energies)

        if total_energy == 0.0:
            return 0.0

        return float(np.max(energies) / total_energy)

    @staticmethod
    def _compute_packet_approximate_energy(packet_leaves: list[np.ndarray]) -> float:
        """Compute the energy of the first wavelet packet leaf."""
        if len(packet_leaves) == 0:
            return 0.0

        return float(np.sum(np.square(packet_leaves[0])))

    @staticmethod
    def _compute_packet_detail_energy(packet_leaves: list[np.ndarray]) -> float:
        """Compute the total energy of all wavelet packet detail leaves."""
        if len(packet_leaves) <= 1:
            return 0.0

        return float(
            sum(
                np.sum(np.square(leaf))
                for leaf in packet_leaves[1:]
            )
        )

    @staticmethod
    def _compute_relative_wavelet_packet_energy(
        packet_leaves: list[np.ndarray],
    ) -> float:
        """Compute the maximum relative energy among wavelet packet leaves."""
        if len(packet_leaves) == 0:
            return 0.0

        energies = np.asarray(
            [
                np.sum(np.square(leaf))
                for leaf in packet_leaves
            ],
            dtype=float,
        )
        total_energy = np.sum(energies)

        if total_energy == 0.0:
            return 0.0

        return float(np.max(energies) / total_energy)

    def compute(self) -> SignalWaveletAnalysisResult:
        """
        Compute wavelet coefficients, wavelet packet leaves, and derived metrics.
        """
        level = self._resolve_level()

        coeffs = pywt.wavedec(self.x, self.params.wavelet, level=level)

        wp = pywt.WaveletPacket(
            data=self.x,
            wavelet=self.params.wavelet,
            mode="symmetric",
            maxlevel=level,
        )
        packet_leaves = [
            node.data
            for node in wp.get_level(level, order="freq")
        ]

        approximate_energy = self._compute_approximate_energy(coeffs)
        detail_energy = self._compute_detail_energy(coeffs)
        relative_wavelet_energy = self._compute_relative_wavelet_energy(coeffs)

        packet_approximate_energy = self._compute_packet_approximate_energy(
            packet_leaves
        )
        packet_detail_energy = self._compute_packet_detail_energy(packet_leaves)
        relative_wavelet_packet_energy = self._compute_relative_wavelet_packet_energy(
            packet_leaves
        )

        return SignalWaveletAnalysisResult(
            coeffs=coeffs,
            packet_leaves=packet_leaves,
            approximate_energy=approximate_energy,
            detail_energy=detail_energy,
            relative_wavelet_energy=relative_wavelet_energy,
            packet_approximate_energy=packet_approximate_energy,
            packet_detail_energy=packet_detail_energy,
            relative_wavelet_packet_energy=relative_wavelet_packet_energy,
        )