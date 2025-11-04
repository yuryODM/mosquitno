import sys
import numpy as np
from mic_array import MicArray
from pixel_ring import pixel_ring

RATE = 16000
CHANNELS = 4           # raw mic channels we use
FRAME_MS = 10          # ms per analysis frame
DOA_FRAMES = 200       # ms per DoA calculation
THRESHOLD = 0.01       # mosquito detection threshold


def is_mosquito(chunk, rate, freq_min=85, freq_max=150, threshold=THRESHOLD):
    """Detect if mosquito frequency is present in the audio chunk."""
    fft = np.fft.rfft(chunk)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(chunk), 1 / rate)

    band_energy = fft_magnitude[(freqs >= freq_min) & (freqs <= freq_max)].sum()
    total_energy = fft_magnitude.sum()

    if total_energy == 0:
        return False

    return (band_energy / total_energy) > threshold


def main():
    mosquito_count = 0
    chunks = []
    doa_chunks = int(DOA_FRAMES / FRAME_MS)

    try:
        with MicArray(RATE, channels=CHANNELS, chunk_size=int(RATE * FRAME_MS / 1000)) as mic:
            for chunk in mic.read_chunks():
                # Use only the first raw channel for mosquito detection
                if is_mosquito(chunk[mic.mic_indices[0]::mic.total_channels], RATE):
                    mosquito_count += 1
                    sys.stdout.write('1')
                else:
                    sys.stdout.write('0')
                sys.stdout.flush()

                chunks.append(chunk)
                if len(chunks) == doa_chunks:
                    if mosquito_count > (doa_chunks / 2):
                        frames = np.concatenate(chunks)
                        direction = mic.get_direction(frames)

                        if direction is not None:
                            pixel_ring.set_direction(direction)
                            print('\nMosquito detected at direction: {:.0f}°'.format(direction))
                        else:
                            pixel_ring.off()
                            print('\nMosquito detected, but direction unknown')

                    mosquito_count = 0
                    chunks = []

    except KeyboardInterrupt:
        pass

    pixel_ring.off()


if __name__ == '__main__':
    main()
