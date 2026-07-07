from __future__ import annotations

import mne
from asrpy import ASR

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class ASRStep(PreprocessingStep):
    """
    ASR preprocessing step with calibration caching.

    Rationale
    ---------
    ``ASR.fit(...)`` is often the most expensive part of the process. This
    implementation allows the model to be:
    - calibrated once through ``prepare(...)``;
    - reused afterwards on the same subject or recording.
    """

    def __init__(
        self,
        cutoff: float = 4.0,
        blocksize: int = 100,
        win_len: float = 1.0,
        win_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        max_bad_chans: float = 0.1,
        method: str = "euclid",
        enable_cache: bool = True,
    ):
        self._cutoff = cutoff
        self._blocksize = blocksize
        self._win_len = win_len
        self._win_overlap = win_overlap
        self._max_dropout_fraction = max_dropout_fraction
        self._min_clean_fraction = min_clean_fraction
        self._max_bad_chans = max_bad_chans
        self._method = method
        self._enable_cache = enable_cache

        self._asr_cache: dict[str, ASR] = {}

    @property
    def name(self) -> str:
        """Return the preprocessing step name."""
        return "asr"

    @property
    def params(self) -> dict:
        """Return the ASR configuration parameters."""
        return {
            "cutoff": self._cutoff,
            "blocksize": self._blocksize,
            "win_len": self._win_len,
            "win_overlap": self._win_overlap,
            "max_dropout_fraction": self._max_dropout_fraction,
            "min_clean_fraction": self._min_clean_fraction,
            "max_bad_chans": self._max_bad_chans,
            "method": self._method,
            "enable_cache": self._enable_cache,
        }

    def _build_asr(self, sfreq: float) -> ASR:
        """Build an ASR model for a given sampling frequency."""
        return ASR(
            sfreq=sfreq,
            cutoff=self._cutoff,
            blocksize=self._blocksize,
            win_len=self._win_len,
            win_overlap=self._win_overlap,
            max_dropout_fraction=self._max_dropout_fraction,
            min_clean_fraction=self._min_clean_fraction,
            max_bad_chans=self._max_bad_chans,
            method=self._method,
        )

    def _get_cache_key(self, eeg_data: EEGData) -> str:
        """Build the cache key associated with an EEG object and ASR settings."""
        return f"asr:{eeg_data.cache_key}:{self._cutoff}:{self._blocksize}:{self._win_len}:{self._win_overlap}:{self._max_dropout_fraction}:{self._min_clean_fraction}:{self._max_bad_chans}:{self._method}"

    def prepare(self, eeg_data: EEGData) -> None:
        """Calibrate ASR once for the given EEG object when caching is enabled."""
        if not self._enable_cache:
            return

        cache_key = self._get_cache_key(eeg_data)

        if cache_key in self._asr_cache:
            return

        with eeg_data.loaded() as raw:
            asr = self._build_asr(float(raw.info["sfreq"]))
            asr.fit(raw)
            self._asr_cache[cache_key] = asr

    def clear_cache(self) -> None:
        """Clear all cached ASR calibrations."""
        self._asr_cache.clear()

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        """Apply ASR correction to an MNE Raw object."""
        if self._enable_cache and eeg_data is not None:
            cache_key = self._get_cache_key(eeg_data)
            asr = self._asr_cache.get(cache_key)

            if asr is None:
                asr = self._build_asr(float(raw.info["sfreq"]))
                asr.fit(raw)
                self._asr_cache[cache_key] = asr

            return asr.transform(raw)

        asr = self._build_asr(float(raw.info["sfreq"]))
        asr.fit(raw)

        return asr.transform(raw)