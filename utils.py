import warnings
from datetime import datetime
from pathlib import Path

import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from stimulus import StimulusParameters

_STIMULUS_FN = "stimulus.wav"
_RECORDED_FN = "recorded.wav"
_RIR_FN = "rir.wav"
_RIR_NONLINEAR_FN = "rir_nonlinear.wav"
_PARAMETER_FN = "parameters.json"
_WAV_SUBTYPE = "PCM_24"


# --------------------------
def record(testsignal: npt.NDArray, fs: int, number_of_recording_channels: int, device: int):

    # Start the recording
    recorded = sd.playrec(
        testsignal, samplerate=fs, channels=number_of_recording_channels, device=device, dtype="float64"
    )
    sd.wait()

    return recorded


# --------------------------
def save_files(  # noqa: PLR0913
    output_dir: Path,
    stimulus_signal: npt.NDArray,
    recorded: npt.NDArray,
    rir: npt.NDArray,
    rir_nonlinear: npt.NDArray,
    parameters: StimulusParameters,
):
    timestamp_str = f"{int(datetime.now().timestamp())}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        warnings.warn(f"Directory {output_dir} is not empty. Files are timestamped.")

    parameters.save_to_json(_format_with_timestamp(output_dir, timestamp_str, _PARAMETER_FN))
    for sig, fn in zip(
        [stimulus_signal, recorded, rir, rir_nonlinear],
        [_STIMULUS_FN, _RECORDED_FN, _RIR_FN, _RIR_NONLINEAR_FN],
    ):
        sf.write(
            _format_with_timestamp(output_dir, timestamp_str, fn),
            sig,
            parameters.fs,
            subtype=_WAV_SUBTYPE,
        )


def _format_with_timestamp(output_dir: Path, timestamp: str, fn: str):
    return output_dir / f"{timestamp}-{fn}"
