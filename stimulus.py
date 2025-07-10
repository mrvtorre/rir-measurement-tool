import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import pi
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy import exp, log, sin
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey

ERROR_EPS = 0.001


class StimulusType(StrEnum):
    """Enumeration of supported stimulus types for signal generation."""

    SINESWEEP = "sine_sweep"


@dataclass
class StimulusParameters:
    """
    Data class for storing parameters used in stimulus signal generation.

    Attributes
    ----------
    fs : int
        Sampling frequency in Hz.
    duration_seconds : float
        Duration of the stimulus in seconds.
    amplitude : float
        Amplitude of the stimulus signal.
    silence_at_start_seconds : float
        Duration of silence at the start in seconds.
    silence_at_end_seconds : float
        Duration of silence at the end in seconds.
    sweep_range : tuple
        Frequency range (start, end) for the sweep in Hz.
    repetitions : int
        Number of repetitions of the stimulus.
    """

    fs: int = 48000
    duration_seconds: float = 15.0
    amplitude: float = 0.7
    silence_at_start_seconds: float = 1.0
    silence_at_end_seconds: float = 1.0
    sweep_range: tuple = (10, 20000)
    repetitions: int = 1

    def save_to_json(self, fn: Path):
        """
        Save the stimulus parameters to a JSON file.

        Parameters
        ----------
        fn : Path
            The file path where the parameters will be saved.
        """
        with open(fn, "w", encoding="utf-8") as fp:
            json.dump(asdict(self), fp)


class Stimulus:
    """Wrapper class around several stimuli."""

    def __init__(self, stimulus_type: StimulusType):

        self.stimulus_type = stimulus_type
        self.Lp = 0
        self.signal = np.array([0])
        self.invfilter = np.array([0])
        self.parameters = None

    def generate(self, parameters: StimulusParameters):
        """Generate a stimulus."""
        self.parameters = parameters

        if self.stimulus_type == StimulusType.SINESWEEP:

            f1 = np.max((parameters.sweep_range[0], 1))  # start of sweep in Hz.
            if parameters.sweep_range[1] == 0:
                f2 = parameters.fs // 2  # end of sweep in Hz. Sweep till Nyquist to avoid ringing
            else:
                f2 = parameters.sweep_range[1]

            w1 = 2 * pi * f1 / parameters.fs  # start of sweep in rad/sample
            w2 = 2 * pi * f2 / parameters.fs  # end of sweep in rad/sample

            n_samples = int(parameters.duration_seconds * parameters.fs)
            sine_sweep = np.zeros(shape=(n_samples, 1))
            taxis = np.arange(0, n_samples, 1) / (n_samples - 1)

            # for exponential sine sweeping
            lw = log(w2 / w1)
            sine_sweep = parameters.amplitude * sin(w1 * (n_samples - 1) / lw * (exp(taxis * lw) - 1), dtype="float64")

            # Find the last zero crossing to avoid the need for fadeout
            # Comment the whole block to remove this
            k = np.flipud(sine_sweep)
            error = 1
            counter = 0
            while error > ERROR_EPS:
                error = np.abs(k[counter])
                counter = counter + 1

            k = k[counter::]
            sine_sweep_hat = np.flipud(k)
            sine_sweep = np.zeros(shape=(n_samples,))
            sine_sweep[0 : sine_sweep_hat.shape[0]] = sine_sweep_hat

            # the convolutional inverse
            envelope = (w2 / w1) ** (-taxis)
            # Holters2009, Eq.(9)
            invfilter = np.flipud(sine_sweep) * envelope
            scaling = pi * n_samples * (w1 / w2 - 1) / (2 * (w2 - w1) * log(w1 / w2)) * (w2 - w1) / pi
            # Holters2009, Eq.10

            # fade-in window. Fade out removed because causes ringing - cropping at zero cross instead
            taper_start = tukey(n_samples, 0)
            taper_window = np.ones(shape=(n_samples,))
            taper_window[0 : int(n_samples / 2)] = taper_start[0 : int(n_samples / 2)]
            sine_sweep = sine_sweep * taper_window

            # Final excitation including repetition and pauses
            sine_sweep = np.expand_dims(sine_sweep, axis=1)
            zerostart = np.zeros(shape=(int(parameters.silence_at_start_seconds * parameters.fs), 1))
            zeroend = np.zeros(shape=(int(parameters.silence_at_end_seconds * parameters.fs), 1))
            sine_sweep = np.concatenate((np.concatenate((zerostart, sine_sweep), axis=0), zeroend), axis=0)
            sine_sweep = np.transpose(np.tile(np.transpose(sine_sweep), parameters.repetitions))

            # Set the attributes
            self.Lp = int(
                (parameters.silence_at_start_seconds + parameters.silence_at_end_seconds + parameters.duration_seconds)
                * parameters.fs
            )
            self.invfilter = invfilter / parameters.amplitude**2 / scaling
            self.signal = sine_sweep
        else:
            raise NameError(f"Excitation type {self.stimulus_type.value} not implemented")

    def deconvolve(self, system_output: npt.NDArray, parameters: StimulusParameters):
        """Deconvolve a system_output in order to return the RIRs."""
        if self.parameters is None or self.parameters != parameters:
            raise ValueError("Parameters should be the same")
        if self.stimulus_type == StimulusType.SINESWEEP:
            n_ch = system_output.shape[1]
            tmplen = self.invfilter.shape[0] + self.Lp - 1
            rirs = np.zeros(shape=(tmplen, n_ch))

            for idx in range(0, n_ch):
                # current_channel = system_output[0:self.repetitions*self.Lp,idx]
                current_channel = system_output[:, idx]
                # rirs[:,idx] = fftconvolve(self.invfilter,current_channel);

                # Average over the repetitions - DEPRECATED. Should not be done.
                sig_reshaped = current_channel.reshape((parameters.repetitions, self.Lp))
                sig_avg = np.mean(sig_reshaped, axis=0)

                # Deconvolution
                rirs[:, idx] = fftconvolve(self.invfilter, sig_avg)
            return rirs
        else:
            raise NameError(f"Excitation type {self.stimulus_type.value} not implemented")


# End of class definition
# ===========================================================================
# ===========================================================================
# NON-CLASS FUNCTIONS


def test_deconvolution(parameters: StimulusParameters):

    stimulus_type = StimulusType.SINESWEEP
    fs = parameters.fs
    duration = parameters.duration_seconds
    repetitions = parameters.repetitions
    silence_at_start = parameters.silence_at_start_seconds

    if repetitions > 1:
        raise NameError(
            "Synchronous time averaging is not recommended for exponential sweeps. A suitable averaging method is not"
            " implemented. Please use a single long sine sweep (e.g. 15 sec.)"
        )

    # Create a test signal object, and generate the excitation
    test_stimulus = Stimulus(stimulus_type)
    test_stimulus.generate(parameters)
    deltapeak = test_stimulus.deconvolve(test_stimulus.signal, parameters)
    startid = int(duration * fs + silence_at_start * fs - 150)
    deltapeak = deltapeak[startid : startid + 300]

    return deltapeak
