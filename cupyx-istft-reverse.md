cupyx.scipy.signal.istft (reversa)
==================================

Assinatura
----------
cupyx.scipy.signal.istft(
    Zxx,
    fs=1.0,
    window='hann',
    nperseg=None,
    noverlap=None,
    nfft=None,
    input_onesided=True,
    boundary=True,
    time_axis=-1,
    freq_axis=-2,
    scaling='spectrum'
)

Perform the inverse Short Time Fourier transform (iSTFT).

Parameters
----------
Zxx (array_like)
    STFT of the signal to be reconstructed. If a purely real array is passed,
    it will be cast to a complex data type.

fs (float, optional)
    Sampling frequency of the time series. Defaults to 1.0.

window (str or tuple or array_like, optional)
    Desired window to use. If window is a string or tuple, it is passed to
    get_window to generate the window values, which are DFT-even by default.
    See get_window for a list of windows and required parameters. If window is
    array_like it will be used directly as the window and its length must be
    nperseg. Defaults to a Hann window. Must match the window used to generate
    the STFT for faithful inversion.

nperseg (int, optional)
    Number of data points corresponding to each STFT segment. This parameter
    must be specified if the number of data points per segment is odd, or if
    the STFT was padded via nfft > nperseg. If None, the value depends on the
    shape of Zxx and input_onesided. If input_onesided is True,
    nperseg=2*(Zxx.shape[freq_axis] - 1). Otherwise, nperseg=Zxx.shape[freq_axis].
    Defaults to None.

noverlap (int, optional)
    Number of points to overlap between segments. If None, half of the segment
    length. Defaults to None. When specified, the COLA constraint must be met
    (see Notes below), and should match the parameter used to generate the STFT.
    Defaults to None.

nfft (int, optional)
    Number of FFT points corresponding to each STFT segment. This parameter
    must be specified if the STFT was padded via nfft > nperseg. If None, the
    default values are the same as for nperseg, detailed above, with one
    exception: if input_onesided is True and
    nperseg==2*Zxx.shape[freq_axis] - 1, nfft also takes on that value. This case
    allows the proper inversion of an odd-length unpadded STFT using nfft=None.
    Defaults to None.

input_onesided (bool, optional)
    If True, interpret the input array as one-sided FFTs, such as is returned by
    stft with return_onesided=True and numpy.fft.rfft. If False, interpret the
    input as a two-sided FFT. Defaults to True.

boundary (bool, optional)
    Specifies whether the input signal was extended at its boundaries by
    supplying a non-None boundary argument to stft. Defaults to True.

time_axis (int, optional)
    Where the time segments of the STFT is located; the default is the last axis
    (i.e. axis=-1).

freq_axis (int, optional)
    Where the frequency axis of the STFT is located; the default is the
    penultimate axis (i.e. axis=-2).

scaling ({'spectrum', 'psd'})
    The default 'spectrum' scaling allows each frequency line of Zxx to be
    interpreted as a magnitude spectrum. The 'psd' option scales each line to a
    power spectral density - it allows to calculate the signal's energy by
    numerically integrating over abs(Zxx)**2.

Returns
-------
t (ndarray)
    Array of output data times.

x (ndarray)
    iSTFT of Zxx.

Notes
-----
In order to enable inversion of an STFT via the inverse STFT with istft, the
signal windowing must obey the constraint of "nonzero overlap add" (NOLA).

An STFT which has been modified (via masking or otherwise) is not guaranteed to
correspond to an exactly realizible signal. This function implements the iSTFT
via the least-squares estimation algorithm detailed in [2], which produces a
signal that minimizes the mean squared error between the STFT of the returned
signal and the modified STFT.

References
----------
[1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck "Discrete-Time Signal Processing", Prentice Hall, 1999.
[2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from Modified Short-Time Fourier Transform", IEEE 1984,
    10.1109/TASSP.1984.1164317

