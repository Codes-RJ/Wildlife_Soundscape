import numpy as np

from localization.filtering import bandpass_filter


def test_bandpass_attenuates_out_of_band_tone():
    fs = 48_000
    t = np.arange(fs // 2) / fs
    inside = np.sin(2 * np.pi * 2000 * t)
    outside = np.sin(2 * np.pi * 50 * t)
    y_inside = bandpass_filter(inside, sample_rate=fs, low_hz=200, high_hz=12000, order=4)
    y_outside = bandpass_filter(outside, sample_rate=fs, low_hz=200, high_hz=12000, order=4)
    assert np.sqrt(np.mean(y_inside**2)) > 5 * np.sqrt(np.mean(y_outside**2))
