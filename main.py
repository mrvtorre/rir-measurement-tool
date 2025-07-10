# ================================================================
# Room impulse response measurement with an exponential sine sweep
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# ================================================================

import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

import _parseargs as parse

# modules from this software
import stimulus as stim
import utils


def main():
    # --- Parse command line arguments and check defaults
    flag_defaults_initialized = parse._checkdefaults()
    args = parse._parse()
    parse._defaults(args)
    # -------------------------------

    if flag_defaults_initialized:
        if args.listdev:
            print(sd.query_devices())
            sd.check_input_settings()
            sd.check_output_settings()
            print("Default input and output device: ", sd.default.device)

        elif args.defaults:
            aa = np.load("_data/defaults.npy", allow_pickle=True).item()
            for i in aa:
                print(i + " => " + str(aa[i]))

        elif args.setdev:

            sd.default.device[0] = args.inputdevice
            sd.default.device[1] = args.outputdevice
            sd.check_input_settings()
            sd.check_output_settings()
            print(sd.query_devices())
            print("Default input and output device: ", sd.default.device)
            print("Sucessfully selected audio devices. Ready to record.")
            parse._defaults(args)

        elif args.test:

            deltapeak = stim.test_deconvolution(args)
            plt.plot(deltapeak)
            plt.show()

        else:

            # Create a test signal object, and generate the excitation
            test_stimulus = stim.Stimulus("sinesweep", args.fs)
            test_stimulus.generate(
                args.fs,
                args.duration,
                args.amplitude,
                args.reps,
                args.startsilence,
                args.endsilence,
                args.sweeprange,
            )

            # Record
            recorded = utils.record(
                test_stimulus.signal,
                args.fs,
                args.inputChannelMap,
                args.outputChannelMap,
            )

            # Deconvolve
            rir = test_stimulus.deconvolve(recorded)

            # Truncate
            len_rir = 1.2
            start_id = test_stimulus.signal.shape[0] - args.endsilence * args.fs - 1
            end_id = start_id + int(len_rir * args.fs)
            # save some more samples before linear part to check for nonlinearities
            start_id_to_save = start_id - int(args.fs / 2)
            rir_to_save = rir[start_id_to_save:end_id, :]
            rir = rir[start_id:end_id, :]

            # Save recordings and rirs
            utils.saverecording(rir, rir_to_save, test_stimulus.signal, recorded, args.fs)


if __name__ == "__main__":
    main()
