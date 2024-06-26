import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa as lc
from torchaudio.transforms import GriffinLim


def make_spectrogram(
    audio: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 128,
    win_length: int = 256,
    log_transform: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Create a spectrogram from an audio sequence

    :param audio: input audio sequence
    :type audio: np.ndarray
    :param n_fft: nfft, defaults to 256
    :type n_fft: int, optional
    :param hop_length: hop_length, defaults to 128
    :type hop_length: int, optional
    :param win_length: win_length, defaults to 256
    :type win_length: int, optional
    :param log_transform: whether to apply log transform, defaults to True
    :type log_transform: bool, optional
    :param normalize: whether to normalize the spectrogram with min-max, defaults to False
    :type normalize: bool, optional

    :return: 2D numpy array of shape (nfft//2, timeframes)

    Usage:
    >>> import numpy as np
    >>> audio = np.random.rand(1000)
    >>> make_spectrogram(audio).shape
    (129, 8)
    """
    spect = lc.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
    )

    if log_transform:
        spect = np.log1p(np.abs(spect))

    if normalize:
        spect = (spect - spect.min()) / (spect.max() - spect.min())

    return spect


def rescale_spectrogram(spect: np.ndarray) -> np.ndarray:
    """Just to make a brighter spectrogram"""
    if np.min(spect) < 0:
        spect -= np.min(spect)
    spect /= np.max(spect)
    return np.log(spect + 0.01)


def plot_spectrogram(
    spect: np.ndarray,
    figsize: tuple[int, int] = (20, 15),
    rescale: bool = True,
    cmap: str = "gray",
) -> plt.Figure:
    """Plot a spectrogram.

    :param spect: input spectrogram
    :type spect: np.ndarray
    :param figsize: figure size, defaults to (20, 15)
    :type figsize: tuple, optional
    :param rescale: whether to rescale the spectrogram, defaults to True
    :type rescale: bool, optional
    :param cmap: colormap, defaults to "gray"
    :type cmap: str, optional

    :return: figure
    :rtype: plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    if rescale:
        spect = rescale_spectrogram(spect)
    ax.imshow(spect, origin="lower", cmap=cmap)
    return fig


def plot_spectrograms(
    spectrograms: list[np.ndarray],
    figsize: tuple[int, int] = (20, 15),
    rescale: bool = True,
    cmap: str = "gray",
) -> plt.Figure:
    """Plot a list of spectrograms.

    :param spectrograms: input spectrograms
    :type spectrograms: List[np.ndarray]
    :param figsize: figure size, defaults to (20, 15)
    :type figsize: tuple, optional
    :param rescale: whether to rescale the spectrogram, defaults to True
    :type rescale: bool, optional
    :param cmap: colormap, defaults to "gray"
    :type cmap: str, optional

    :return: figure
    :rtype: plt.Figure
    """
    n = len(spectrograms)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    for i, ax in enumerate(axes):
        if rescale:
            spect = rescale_spectrogram(spectrograms[i])
        ax.imshow(spect, origin="lower", cmap=cmap)
    return fig


def play_audio_sounddevice(audio: np.ndarray, sr: int) -> None:
    """Play audio using sounddevice.

    :param audio: input audio sequence
    :type audio: np.ndarray
    :param sr: sample rate
    :type sr: int
    """
    import sounddevice as sd

    sd.play(audio, samplerate=sr)
    sd.wait()


def random_time_crop_spectrogram(
    spectrogram: np.array,
    crop_length: int,
) -> np.array:
    """Randomly crop a spectrogram in the time dimension.

    :param spectrogram: input spectrogram, shape (nfft//2, timeframes)
    :type spectrogram: np.array
    :param crop_length: length of the crop
    :type crop_length: int
    :return: cropped spectrogram
    :rtype: np.array
    """
    n = spectrogram.shape[1]
    if n < crop_length:
        raise ValueError(
            "Crop length should be less than the time dimension of the spectrogram"
        )
    start = np.random.randint(0, n - crop_length)
    return spectrogram[:, start : start + crop_length]


def make_audio_from_spectrogram(
    spectrogram: np.ndarray | torch.Tensor, n_fft: int = 256, hop_length: int = 128
) -> np.ndarray:
    """Create audio from a spectrogram using Griffin-Lim algorithm.

    :param spectrogram: input audio sequence
    :type spectrogram: np.ndarray | torch.Tensor
    :param n_fft: nfft (window size), defaults to 256
    :type n_fft: int
    :param hop_length: hop_length (frame shift), defaults to 128
    :type hop_length: int
    :return: audio sequence
    :rtype: np.ndarray
    """
    if isinstance(spectrogram, np.ndarray):
        spectrogram = torch.from_numpy(spectrogram)

    # make sure shape of audio is [nfft//2, timeframes]
    assert spectrogram.shape[0] == n_fft // 2 + 1, "Invalid shape of spectrogram"
    assert spectrogram.dim() == 2, "Invalid shape of spectrogram"

    griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length)
    return griffin_lim(spectrogram)
