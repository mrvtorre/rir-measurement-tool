import warnings
from datetime import datetime
from pathlib import Path

import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from stimulus import StimulusParameters

_STIMULUS_FN = "stimulus.wav"
_RIR_FN = "rir.wav"
_RIR_NONLINEAR_FN = "rir_nonlinear.json"
_PARAMETER_FN = "parameters.json"


# --------------------------
def record(testsignal: npt.NDArray, fs: int, number_of_playback_channels: int, device: int):

    # Start the recording
    recorded = sd.playrec(
        testsignal, samplerate=fs, channels=number_of_playback_channels, device=device, dtype="float64"
    )
    sd.wait()

    return recorded


# --------------------------
def save_files(
    output_dir: Path,
    stimulus_signal: npt.NDArray,
    rir: npt.NDArray,
    rir_nonlinear: npt.NDArray,
    parameters: StimulusParameters,
):
    timestamp_str = f"{int(datetime.now().timestamp())}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        warnings.warn(f"Directory {output_dir} is not empty. Files are timestamped.")

    sf.write(_format_with_timestamp(output_dir, timestamp_str, _STIMULUS_FN), stimulus_signal, parameters.fs)
    sf.write(_format_with_timestamp(output_dir, timestamp_str, _RIR_FN), rir, parameters.fs)
    sf.write(_format_with_timestamp(output_dir, timestamp_str, _RIR_NONLINEAR_FN), rir_nonlinear, parameters.fs)
    parameters.save_to_json(_format_with_timestamp(output_dir, timestamp_str, _PARAMETER_FN))


def _format_with_timestamp(output_dir: Path, timestamp: str, fn: str):
    return output_dir / f"{timestamp}-{fn}"
