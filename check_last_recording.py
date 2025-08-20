# ================================================================
# Utility file to visualize the result from the last recording
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# Modified by:               Mike Vantorre
# ================================================================


import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.signal import spectrogram

from stimulus import StimulusParameters


def check_last_recording(test_signal: NDArray, recorded: NDArray, rir: NDArray, parameters: StimulusParameters):
    fs = parameters.fs

    # THE ROOM IMPULSE RESPONSES
    maxval = np.max(rir)
    minval = np.min(rir)

    # Plot them as subplots
    n_signals = rir.shape[-1]
    nrows = n_signals // 2 if n_signals % 2 == 0 else n_signals // +1
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for idx in range(n_signals):
        axs[idx // 2, idx % 2].plot(rir[:, idx], linewidth=0.5)
        axs[idx // 2, idx % 2].set_ylim((minval + 0.05 * minval, maxval + 0.05 * maxval))
        axs[idx // 2, idx % 2].set_title("rir Microphone " + str(idx + 1))
    fig.tight_layout()

    fig = plt.figure(figsize=(16, 10))
    plt.plot(test_signal, color="r", linewidth=0.5)
    plt.title("Computer-generated test signal")

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for idx in range(n_signals):
        axs[idx // 2, idx % 2].plot(recorded[:, idx], color="b", linewidth=0.5)
        axs[idx // 2, idx % 2].set_title("Recording at Microphone " + str(idx + 1))
    fig.tight_layout()

    nperseg = 2**11
    sweepnfft = nperseg
    noverlap = int(0.5 * nperseg)
    faxis = np.linspace(0, fs, sweepnfft)
    faxis = faxis[0 : int(sweepnfft / 2)]

    _, _, spectest = spectrogram(
        np.squeeze(test_signal) + 1e-12,
        fs=fs,
        nperseg=nperseg,
        nfft=sweepnfft,
        noverlap=noverlap,
        scaling="spectrum",
    )
    spectest = spectest[1::, :] + 1e-12

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=4,
        figsize=(16, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    col = 0
    for idx in range(recorded.shape[1]):
        col %= 4

        _, _, specrec = spectrogram(
            recorded[:, idx] + 1e-12,
            fs=fs,
            nperseg=nperseg,
            nfft=sweepnfft,
            noverlap=noverlap,
            scaling="spectrum",
        )

        specrec = specrec[1::, :] + 1e-12

        taxis = np.arange(0, spectest.shape[1], 1)

        ax = axs[idx // 2, col]
        p = ax.pcolormesh(taxis, faxis, 20 * np.log10(spectest), cmap="hot", vmin=-130)
        ax.set_yscale("log")
        ax.set_ylim((20, 20000))
        fig.colorbar(p, ax=ax, orientation="horizontal", fraction=0.06)

        col += 1
        ax = axs[idx // 2, col]
        p = ax.pcolormesh(taxis, faxis, 20 * np.log10(specrec), cmap="hot", vmin=-130)
        ax.set_yscale("log")
        ax.set_ylim((20, 20000))
        fig.colorbar(p, ax=ax, orientation="horizontal", fraction=0.06)

        pos1 = axs[idx // 2, col - 1].get_position()
        pos2 = axs[idx // 2, col].get_position()
        x = (pos1.x0 + pos2.x1) / 2
        y = pos1.y1 + 0.002

        fig.text(x, y, f"Microphone {idx}", ha="center", va="bottom", fontsize=10)

        col += 1
    fig.suptitle("Test signal vs. recorded signal.")

    plt.show()
