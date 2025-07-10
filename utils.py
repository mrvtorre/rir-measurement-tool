import os

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wavwrite


# --------------------------
def record(testsignal, fs, input_channels, output_channels):

    sd.default.samplerate = fs
    sd.default.dtype = "float32"
    print("Input channels:", input_channels)
    print("Output channels:", output_channels)

    # Start the recording
    recorded = sd.playrec(
        testsignal,
        samplerate=fs,
        input_mapping=input_channels,
        output_mapping=output_channels,
    )
    sd.wait()

    return recorded


# --------------------------
def saverecording(rir, rir_to_save, testsignal, recorded, fs):

    dirflag = False
    counter = 1
    dirname = "recorded/newrir1"
    while dirflag:
        if os.path.exists(dirname):
            counter = counter + 1
            dirname = "recorded/newrir" + str(counter)
        else:
            os.mkdir(dirname)
            dirflag = True

    # Saving the rirs and the captured signals
    np.save(dirname + "/rir.npy", rir)
    np.save(dirname + "/rirac.npy", rir_to_save)
    wavwrite(dirname + "/sigtest.wav", fs, testsignal)

    for idx in range(recorded.shape[1]):
        wavwrite(dirname + "/sigrec" + str(idx + 1) + ".wav", fs, recorded[:, idx])
        wavwrite(dirname + "/rir" + str(idx + 1) + ".wav", fs, rir[:, idx])

    # Save in the recorded/lastRecording for a quick check
    np.save("recorded/lastRecording/rir.npy", rir)
    np.save("recorded/lastRecording/rirac.npy", rir_to_save)
    wavwrite("recorded/lastRecording/sigtest.wav", fs, testsignal)
    for idx in range(recorded.shape[1]):
        wavwrite("sigrec" + str(idx + 1) + ".wav", fs, recorded[:, idx])
        wavwrite(dirname + "/rir" + str(idx + 1) + ".wav", fs, rir[:, idx])

    print("Success! Recording saved in directory " + dirname)
