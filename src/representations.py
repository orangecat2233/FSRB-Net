from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps
from scipy.signal import stft


REPRESENTATIONS = ("cwt", "stft", "chirplet", "rp", "wsst", "gadf", "mssi")


def _normalize(matrix, low=0.0, high=100.0):
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if low == 0.0 and high == 100.0:
        lo, hi = float(matrix.min()), float(matrix.max())
    else:
        lo, hi = np.percentile(matrix, [low, high])
    if hi <= lo + 1e-12:
        return np.zeros_like(matrix, dtype=np.float32)
    return np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)


def _save_rgb(matrix, output_path, image_size, cmap_name="jet", resample=Image.Resampling.BILINEAR):
    rgb = np.round(colormaps.get_cmap(cmap_name)(_normalize(matrix))[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((image_size, image_size), resample)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def cwt_matrix(signal, sampling_rate, scales=128, wavelet="cmor100-1"):
    import pywt
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    scale_count = int(scales)
    center = pywt.central_frequency(wavelet)
    scale_values = 2 * center * scale_count / np.arange(scale_count, 0, -1)
    coefficients, _ = pywt.cwt(signal, scale_values, wavelet, sampling_period=1.0 / float(sampling_rate))
    return np.abs(coefficients)


def stft_matrix(signal, sampling_rate, window_size=64, overlap=0.5, nfft=None, log_scale=True):
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    window = min(int(window_size), signal.size)
    noverlap = min(int(window * overlap), window - 1)
    _, _, z = stft(signal, fs=float(sampling_rate), window="hann", nperseg=window, noverlap=noverlap, nfft=nfft, boundary=None, padded=False)
    matrix = np.abs(z)
    return np.log1p(matrix) if log_scale else matrix


def chirplet_matrix(signal, sampling_rate, window_size=128, nfft=256, chirp_rates=(-400000, -200000, 0, 200000, 400000)):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = min(int(window_size), signal.size)
    time_axis = (np.arange(n) - (n - 1) / 2) / float(sampling_rate)
    gaussian = np.exp(-0.5 * (time_axis / (n / (6 * sampling_rate))) ** 2)
    matrices = []
    for rate in chirp_rates:
        window = gaussian * np.exp(1j * np.pi * rate * time_axis**2)
        _, _, z = stft(signal, fs=float(sampling_rate), window=window, nperseg=n, noverlap=n // 2, nfft=nfft, boundary=None, padded=False, return_onesided=False)
        matrices.append(np.abs(z[: nfft // 2 + 1]))
    return np.log1p(np.maximum.reduce(matrices))


def rp_matrix(signal, dimension=1, time_delay=1, threshold=None, percentage=10):
    from pyts.image import RecurrencePlot
    x = np.array(signal, dtype=np.float64, order="C", copy=True).reshape(1, -1)
    return RecurrencePlot(dimension=dimension, time_delay=time_delay, threshold=threshold, percentage=percentage, flatten=False).fit_transform(x)[0]


def gadf_matrix(signal, image_size=128, sample_range=(-1, 1), overlapping=False):
    from pyts.image import GramianAngularField
    x = np.array(signal, dtype=np.float64, order="C", copy=True).reshape(1, -1)
    size = min(int(image_size), x.shape[1])
    return GramianAngularField(image_size=size, sample_range=sample_range, method="difference", overlapping=overlapping, flatten=False).fit_transform(x)[0]


def wsst_matrix(signal, sampling_rate, wavelet="morlet", voices_per_octave=32, log_scale=True, power=1.0, freq_min=None, freq_max=None):
    try:
        from ssqueezepy import ssq_cwt
    except ImportError as exc:
        raise ImportError("WSST conversion requires ssqueezepy") from exc
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    x = x - x.mean()
    std = float(x.std())
    if std > 1e-8:
        x = x / std
    tx, _, freqs, _ = ssq_cwt(
        x,
        wavelet=wavelet,
        scales="log-piecewise",
        nv=int(voices_per_octave),
        fs=float(sampling_rate),
        padtype="reflect",
        squeezing="sum",
        maprange="peak",
        astensor=False,
    )
    tx = np.asarray(tx)
    freqs = np.asarray(freqs).reshape(-1)
    mag = np.abs(tx).astype(np.float32)
    if power != 1.0:
        mag = np.power(mag, power)
    if log_scale:
        mag = np.log1p(mag)
    valid = np.isfinite(freqs) & (freqs >= 0)
    if freq_min is not None:
        valid &= freqs >= float(freq_min)
    if freq_max is not None:
        valid &= freqs <= float(freq_max)
    if not np.any(valid):
        raise ValueError("No WSST frequency bins remain after frequency filtering")
    freqs, mag = freqs[valid], mag[valid]
    order = np.argsort(freqs)[::-1]
    return mag[order]


def _minmax(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-8)


def _segment_for_fft(signal, target_length):
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size >= target_length:
        return x[:target_length]
    out = np.zeros(target_length, dtype=np.float32)
    out[:x.size] = x
    return out


def mssi_matrix(signal, m_values=range(3, 10), remove_mean=True):
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if remove_mean:
        x = x - np.mean(x, dtype=np.float64)
    m_values = tuple(int(m) for m in m_values)
    max_length = 2 ** (max(m_values) - 1)
    rows = []
    for m in m_values:
        n = 2**m
        seg = _segment_for_fft(x, n)
        spec = np.abs(np.fft.fft(seg, n=n))[: n // 2]
        spec = _minmax(spec)
        idx = np.floor(np.arange(max_length) * spec.size / max_length).astype(np.int64)
        rows.append(spec[np.clip(idx, 0, spec.size - 1)])
    stacked = np.stack(rows).astype(np.float32)
    if len(m_values) == 7 and max_length == 256:
        matrix = stacked.reshape(56, 32)
    else:
        total = stacked.size
        h = int(round(np.sqrt(total)))
        while h > 1 and total % h:
            h -= 1
        matrix = stacked.reshape(h, total // h)
    return _minmax(matrix)


def make_representation(signal, output_path, method, sampling_rate, image_size=128):
    method = method.lower()
    if method not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {method}")
    if method == "cwt":
        matrix = cwt_matrix(signal, sampling_rate)
    elif method == "stft":
        matrix = stft_matrix(signal, sampling_rate)
    elif method == "chirplet":
        matrix = chirplet_matrix(signal, sampling_rate)
        matrix = np.abs(np.asarray(matrix, dtype=np.float32))
        low, high = np.percentile(matrix, [1.0, 99.0])
        if high > low:
            matrix = np.clip((matrix - low) / (high - low), 0.0, 1.0)
        else:
            matrix = np.zeros_like(matrix, dtype=np.float32)
        rgb = np.round(colormaps.get_cmap("jet")(matrix)[..., :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return
    elif method == "rp":
        matrix = rp_matrix(signal)
    elif method == "wsst":
        matrix = wsst_matrix(signal, sampling_rate)
        low, high = np.percentile(matrix, [1.0, 99.5])
        if high <= low + 1e-8:
            low, high = float(np.min(matrix)), float(np.max(matrix))
        matrix = np.clip((matrix - low) / (high - low + 1e-8), 0.0, 1.0)
        resized = Image.fromarray(matrix.astype(np.float32), mode="F").resize((image_size, image_size), Image.Resampling.BILINEAR)
        matrix = np.asarray(resized, dtype=np.float32)
        rgb = np.round(colormaps.get_cmap("jet")(matrix)[..., :3] * 255).astype(np.uint8)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(output_path)
        return
    elif method == "gadf":
        matrix = gadf_matrix(signal, image_size)
    else:
        matrix = _normalize(mssi_matrix(signal))
        gray = np.round(matrix * 255.0).astype(np.uint8)
        image = Image.fromarray(gray, mode="L").resize((image_size, image_size), Image.Resampling.NEAREST).convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return
    _save_rgb(matrix, output_path, image_size, resample=Image.Resampling.BICUBIC if method == "cwt" else Image.Resampling.BILINEAR)
