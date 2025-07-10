from math import pi

import numpy as np
from numpy import exp, log, sin
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey

ERROR_EPS = 0.001


class Stimulus:
    """Wrapper class around several stimuli."""

    # Constructor
    def __init__(self, stimulus_type, sampling_rate):

        self.stimulus_type = stimulus_type
        self.fs = sampling_rate
        self.repetitions = 0
        self.Lp = []
        self.signal = []
        self.invfilter = []

    # Generate the stimulus and set requred attributes
    def generate(
        self,
        fs,
        duration,
        amplitude,
        repetitions,
        silence_at_start,
        silence_at_end,
        sweeprange,
    ):
        """Generate a stimulus."""
        if self.stimulus_type == "sinesweep":

            f1 = np.max((sweeprange[0], 1))  # start of sweep in Hz.
            if sweeprange[1] == 0:
                f2 = int(fs / 2)  # end of sweep in Hz. Sweep till Nyquist to avoid ringing
            else:
                f2 = sweeprange[1]

            w1 = 2 * pi * f1 / fs  # start of sweep in rad/sample
            w2 = 2 * pi * f2 / fs  # end of sweep in rad/sample

            n_samples = duration * fs
            sine_sweep = np.zeros(shape=(n_samples, 1))
            taxis = np.arange(0, n_samples, 1) / (n_samples - 1)

            # for exponential sine sweeping
            lw = log(w2 / w1)
            sine_sweep = amplitude * sin(w1 * (n_samples - 1) / lw * (exp(taxis * lw) - 1))

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
            zerostart = np.zeros(shape=(silence_at_start * fs, 1))
            zeroend = np.zeros(shape=(silence_at_end * fs, 1))
            sine_sweep = np.concatenate((np.concatenate((zerostart, sine_sweep), axis=0), zeroend), axis=0)
            sine_sweep = np.transpose(np.tile(np.transpose(sine_sweep), repetitions))

            # Set the attributes
            self.Lp = (silence_at_start + silence_at_end + duration) * fs
            self.invfilter = invfilter / amplitude**2 / scaling
            self.repetitions = repetitions
            self.signal = sine_sweep

        else:

            raise NameError("Excitation type not implemented")
            return

    def deconvolve(self, system_output):
        """Deconvolve a system_output in order to return the RIRs."""
        if self.stimulus_type == "sinesweep":

            n_ch = system_output.shape[1]
            tmplen = self.invfilter.shape[0] + self.Lp - 1
            rirs = np.zeros(shape=(tmplen, n_ch))

            for idx in range(0, n_ch):

                # current_channel = system_output[0:self.repetitions*self.Lp,idx]
                current_channel = system_output[:, idx]
                # rirs[:,idx] = fftconvolve(self.invfilter,current_channel);

                # Average over the repetitions - DEPRECATED. Should not be done.
                sig_reshaped = current_channel.reshape((self.repetitions, self.Lp))
                sig_avg = np.mean(sig_reshaped, axis=0)

                # Deconvolution
                rirs[:, idx] = fftconvolve(self.invfilter, sig_avg)

            return rirs

        else:

            raise NameError("Excitation type not implemented")
            return


# End of class definition
# ===========================================================================
# ===========================================================================
# NON-CLASS FUNCTIONS


def test_deconvolution(args):

    stimulus_type = "sinesweep"
    fs = args.fs
    duration = args.duration
    amplitude = args.amplitude
    repetitions = args.reps
    silence_at_start = args.startsilence
    silence_at_end = args.endsilence
    sweeprange = args.sweeprange

    if repetitions > 1:
        raise NameError(
            "Synchronous time averaging is not recommended for exponential sweeps. A suitable averaging method is not"
            " implemented. Please use a single long sine sweep (e.g. 15 sec.)"
        )

    # Create a test signal object, and generate the excitation
    test_stimulus = Stimulus(stimulus_type, fs)
    test_stimulus.generate(fs, duration, amplitude, repetitions, silence_at_start, silence_at_end, sweeprange)
    deltapeak = test_stimulus.deconvolve(test_stimulus.signal)
    startid = duration * fs + silence_at_start * fs - 150
    deltapeak = deltapeak[startid : startid + 300]

    return deltapeak
