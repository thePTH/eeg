from preprocessing.step.base import PreprocessingStep, EEGData
from asrpy import ASR


from asrpy import ASR


class ASRStep(PreprocessingStep):
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
    ):
        self._cutoff = cutoff
        self._blocksize = blocksize
        self._win_len = win_len
        self._win_overlap = win_overlap
        self._max_dropout_fraction = max_dropout_fraction
        self._min_clean_fraction = min_clean_fraction
        self._max_bad_chans = max_bad_chans
        self._method = method

    @property
    def name(self) -> str:
        return "asr"

    @property
    def params(self) -> dict:
        return {
            "cutoff": self._cutoff,
            "blocksize": self._blocksize,
            "win_len": self._win_len,
            "win_overlap": self._win_overlap,
            "max_dropout_fraction": self._max_dropout_fraction,
            "min_clean_fraction": self._min_clean_fraction,
            "max_bad_chans": self._max_bad_chans,
            "method": self._method,
        }

    def transform(self, eeg_data):
        eeg_new = eeg_data.copy()
        raw_original = eeg_new.raw

    
        asr = ASR(
            sfreq=raw_original.info["sfreq"],
            cutoff=self._cutoff,
            blocksize=self._blocksize,
            win_len=self._win_len,
            win_overlap=self._win_overlap,
            max_dropout_fraction=self._max_dropout_fraction,
            min_clean_fraction=self._min_clean_fraction,
            max_bad_chans=self._max_bad_chans,
            method=self._method,
        )

        # Calibration automatique sur segments propres (fait en interne par ASRpy)
        asr.fit(raw_original)

        # Reconstruction du signal
        raw_clean = asr.transform(raw_original)

        
        
        eeg_new._raw = raw_clean


        return eeg_new