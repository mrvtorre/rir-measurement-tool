# ================================================================
# Room impulse response measurement with an exponential sine sweep
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# Modified by:               Mike Vantorre
# ================================================================
from pathlib import Path

import click
import sounddevice as sd
from matplotlib import pyplot as plt

# modules from this software
import stimulus
import utils

_DEFAULT_PARAMS = stimulus.StimulusParameters()
_LEN_RIR_S = 1.2


@click.group()
def cli():
    """Command line interface for RIR measurement using exponential sine sweep."""
    pass


def _list_devices():
    click.echo("Available devices:")
    click.echo(sd.query_devices())
    sd.check_input_settings()
    sd.check_output_settings()
    default_input_device, default_output_device = sd.default.device
    click.echo(f"Default input and output device: [{default_input_device}, {default_output_device}]")
    return default_input_device, default_output_device


def _prompt_for_devices():
    default_input_device, default_output_device = _list_devices()
    input_device = click.prompt("Input device: ", type=int, default=default_input_device, show_default=True)
    output_device = click.prompt("Output device: ", type=int, default=default_output_device, show_default=True)
    return input_device, output_device


@cli.command()
def list_devices():
    """List available audio devices."""
    _list_devices()


@cli.command()
def set_default_devices():
    """Set the default audio input and output devices."""
    input_device, output_device = _prompt_for_devices()
    sd.default.device = [input_device, output_device]
    sd.check_input_settings()
    sd.check_output_settings()
    click.echo("Successfully selected audio devices. Ready to record.")


@cli.command(name="test")
def test_deconv():
    """Test the deconvolution operation with the default arguments."""
    deltapeak = stimulus.test_deconvolution(stimulus.StimulusParameters())
    plt.plot(deltapeak)
    plt.show()


@cli.command()
@click.option("--fs", default=_DEFAULT_PARAMS.fs, type=int, show_default=True)
@click.option("--duration_seconds", default=_DEFAULT_PARAMS.duration_seconds, type=float, show_default=True)
@click.option("--amplitude", default=_DEFAULT_PARAMS.amplitude, type=float, show_default=True)
@click.option(
    "--silence_at_start_seconds", default=_DEFAULT_PARAMS.silence_at_start_seconds, type=float, show_default=True
)
@click.option("--silence_at_end_seconds", default=_DEFAULT_PARAMS.silence_at_end_seconds, type=float, show_default=True)
@click.option("--sweep_range", default=_DEFAULT_PARAMS.sweep_range, type=click.Tuple([float, float]), show_default=True)
@click.option("--number_of_playback_channels", "-n_channels", type=int, prompt=True)
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path))
def measure(  # noqa: PLR0913
    fs: int,
    duration_seconds: float,
    amplitude: float,
    silence_at_start_seconds: float,
    silence_at_end_seconds: float,
    sweep_range: tuple,
    number_of_playback_channels: int,
    output_dir: Path,
):
    """Run the measure script: generate a stimulus signal, perform a recording, and deconvolve into responses."""
    default_input_device, _ = _list_devices()
    device = click.prompt("Device: ", type=int, default=default_input_device, show_default=True)

    parameters = stimulus.StimulusParameters(
        fs=fs,
        duration_seconds=duration_seconds,
        amplitude=amplitude,
        silence_at_start_seconds=silence_at_start_seconds,
        silence_at_end_seconds=silence_at_end_seconds,
        sweep_range=sweep_range,
    )

    click.echo(f"Using following parameters: {parameters}")

    test_stimulus = stimulus.Stimulus(stimulus.StimulusType.SINESWEEP)
    test_stimulus.generate(parameters)

    # Record
    recorded = utils.record(
        test_stimulus.signal,
        parameters.fs,
        number_of_playback_channels,
        device,
    )

    # Deconvolve
    rir = test_stimulus.deconvolve(recorded, parameters)

    # Truncate
    start_id = test_stimulus.signal.shape[0] - parameters.silence_at_end_seconds * parameters.fs - 1
    end_id = start_id + int(_LEN_RIR_S * parameters.fs)
    # save some more samples before linear part to check for nonlinearities
    start_id_to_save = start_id - int(parameters.fs / 2)
    rir_nonlinear = rir[start_id_to_save:end_id, :]
    rir = rir[start_id:end_id, :]

    # Save recordings and rirs
    utils.save_files(output_dir, test_stimulus.signal, rir, rir_nonlinear, parameters)


if __name__ == "__main__":
    cli()
