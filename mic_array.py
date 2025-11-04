import pyaudio
import queue
import threading
import numpy as np
from gcc_phat import gcc_phat
import math

SOUND_SPEED = 343.2

MIC_DISTANCE_4 = 0.08127  # distance between adjacent mics
MAX_TDOA_4 = MIC_DISTANCE_4 / SOUND_SPEED


class MicArray:
    """
    Handles audio input from 6-channel mic array.
    Channel mapping:
        0: processed audio (ignore)
        1-4: raw mic channels (used for DoA)
        5: merged playback (ignore)
    """

    def __init__(self, rate=16000, channels=4, chunk_size=None):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.sample_rate = rate
        self.total_channels = 6           # physical channels from the device
        self.channels = channels          # number of raw mic channels to use
        self.chunk_size = chunk_size if chunk_size else int(rate / 100)
        # map raw mic channels dynamically
        self.mic_indices = list(range(1, 1 + self.channels))

        # Find input device
        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            print(i, dev['name'], dev['maxInputChannels'], dev['maxOutputChannels'])
            if dev['maxInputChannels'] >= max(self.mic_indices) + 1:
                print('Use device:', dev['name'])
                device_index = i
                break

        if device_index is None:
            raise Exception(
                f'Cannot find input device with at least {max(self.mic_indices)+1} channels'
            )

        self.stream = self.pyaudio_instance.open(
            input=True,
            start=False,
            format=pyaudio.paInt16,
            channels=self.total_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
            input_device_index=device_index,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def start(self):
        self.queue.queue.clear()
        self.stream.start_stream()

    def read_chunks(self):
        self.quit_event.clear()
        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break
            frames = np.frombuffer(frames, dtype=np.int16)
            yield frames

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()
        self.queue.put(b'')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if value:
            return False
        self.stop()

    def get_direction(self, buf):
        """
        Compute DoA in degrees [0, 360] using GCC-PHAT.
        Returns None if invalid.
        """
        if self.channels < 2:
            return None  # cannot compute DoA with less than 2 mics

        best_guess = None
        # generate mic pairs (adjacent mics)
        MIC_GROUP = [[i, i + 1] for i in range(self.channels - 1)]
        tau_list = []
        theta_list = []

        for pair in MIC_GROUP:
            ch0 = self.mic_indices[pair[0]]
            ch1 = self.mic_indices[pair[1]]

            try:
                tau, _ = gcc_phat(
                    buf[ch0::self.total_channels],
                    buf[ch1::self.total_channels],
                    fs=self.sample_rate,
                    max_tau=MAX_TDOA_4,
                    interp=1,
                )
                # clamp tau
                tau = max(-MAX_TDOA_4, min(MAX_TDOA_4, tau))
                tau_list.append(tau)
                theta_list.append(math.asin(tau / MAX_TDOA_4) * 180 / math.pi)
            except Exception:
                tau_list.append(0)
                theta_list.append(0)

        if tau_list:
            min_idx = np.argmin(np.abs(tau_list))
            angle = theta_list[min_idx]
            best_guess = (angle + 360) % 360

        return best_guess
