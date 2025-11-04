import pyaudio
import queue
import threading
import numpy as np
from gcc_phat import gcc_phat
import math

SOUND_SPEED = 343.2

MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / SOUND_SPEED


class MicArray:
    def __init__(self, rate=16000, channels=6, chunk_size=None):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()

        self.sample_rate = rate
        self.channels = channels              # total device channels
        self.chunk_size = chunk_size if chunk_size else int(rate / 100)
        self.mic_indices = [1, 2, 3, 4]      # raw mic channels for DOA
        self.total_channels = self.channels  # used for slicing

        # Find input device with enough channels
        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            if dev['maxInputChannels'] >= self.channels:
                print(f"Use device: {dev['name']}")
                device_index = i
                break

        if device_index is None:
            raise Exception(f"Cannot find input device with at least {self.channels} channels")

        # Open stream
        self.stream = self.pyaudio_instance.open(
            input=True,
            start=False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=int(self.sample_rate),
            frames_per_buffer=int(self.chunk_size),
            stream_callback=self._callback,
            input_device_index=device_index
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
            frames = np.frombuffer(frames, dtype='int16')
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
        Compute DOA from 4 raw mics (channels 1-4).
        Returns azimuth in degrees.
        """
        MIC_GROUP_N = 2
        MIC_GROUP = [[1, 3], [2, 4]]  # mic pairs for TDOA

        tau = [0] * MIC_GROUP_N
        theta = [0] * MIC_GROUP_N

        for i, pair in enumerate(MIC_GROUP):
            # Slice each mic from the interleaved buffer
            mic_a = buf[pair[0]::self.total_channels]
            mic_b = buf[pair[1]::self.total_channels]
            tau[i], _ = gcc_phat(mic_a, mic_b, fs=self.sample_rate, max_tau=MAX_TDOA_4, interp=1)
            theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi

        # Combine two theta estimates into a single azimuth
        if np.abs(theta[0]) < np.abs(theta[1]):
            if theta[1] > 0:
                best_guess = (theta[0] + 360) % 360
            else:
                best_guess = (180 - theta[0])
        else:
            if theta[0] < 0:
                best_guess = (theta[1] + 360) % 360
            else:
                best_guess = (180 - theta[1])
            best_guess = (best_guess + 90 + 180) % 360

        # Optional calibration offset
        best_guess = (-best_guess + 120) % 360
        return best_guess
