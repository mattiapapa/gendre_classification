import struct
import numpy as np
import webrtcvad


fs = 16000  # LibriSpeech sampling frequency [Hz]
dur = 1     # Audio duration [s]
win_dur = 0.02
max_attempts = 20


def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def run_vad(data, aggress=2, window_duration=win_dur, samplerate=fs):
    vad = webrtcvad.Vad()
    vad.set_mode(aggress)
    audio = float2pcm(data)
    raw_samples = struct.pack("%dh" % len(audio), *audio)
    samples_per_window = int(window_duration * samplerate)
    number_windows = int(np.floor(len(audio) / samples_per_window))
    bytes_per_sample = 2

    segments = []
    for i in np.arange(number_windows):
        raw_frame = raw_samples[i * bytes_per_sample * samples_per_window:
                                (i + 1) * bytes_per_sample * samples_per_window]
        is_speech = vad.is_speech(raw_frame, sample_rate=samplerate)
        segments.append(dict(
            start=i * samples_per_window,
            stop=(i + 1) * samples_per_window - 1,
            is_speech=is_speech))

    old_bool = segments[0]['is_speech']
    new_start = segments[0]['start']

    long_segments = []
    for i, segment in enumerate(segments):
        new_bool = segment['is_speech']
        if old_bool == new_bool:
            new_stop = segment['stop']
        else:
            long_segments.append(dict(
                start=new_start,
                stop=new_stop,
                is_speech=old_bool))
            new_start = segment['start']
            new_stop = segment['stop']
        old_bool = new_bool
        if i == len(segments) - 1:
            long_segments.append(dict(
                start=new_start,
                stop=new_stop,
                is_speech=old_bool))
    return long_segments