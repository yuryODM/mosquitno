import sys
import time
import numpy as np
from mic_array import MicArray
from pixel_ring import pixel_ring
import logging

RATE = 16000
CHANNELS = 6           # total USB channels from ReSpeaker device
FRAME_MS = 10          # ms per analysis frame
THRESHOLD = 0.15     # mosquito detection threshold

logging.basicConfig(
    level=logging.INFO,                      # minimum level to log
    format='%(asctime)s [%(levelname)s] %(message)s',  # log format
    handlers=[
        logging.FileHandler("mosquito_audio.log"), # save to file
        logging.StreamHandler()              # also print to console
    ]
)

def is_mosquito(chunk, rate, freq_min=650, freq_max=850, threshold=THRESHOLD):
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
    try:
        # Open the mic array with 6 channels (ASR + 4 raw mics + merged)
        chunk_size = int(RATE * FRAME_MS / 1000)
        with MicArray(RATE, channels=CHANNELS, chunk_size=chunk_size) as mic:
            for chunk in mic.read_chunks():
                # Use only the first raw mic channel (channel 1)
                start_time = time.time()  # time spent processing this chunk
                channel_data = chunk[mic.mic_indices[0]::mic.total_channels]  # channel 1, step by total channels
                if is_mosquito(channel_data, RATE):
                    logging.info('Mosquito detected!')

                    # Compute direction immediately using all raw mic channels
                    direction = mic.get_direction(chunk)

                    processing_time = time.time() - start_time 
                    
                    if direction is not None:
                        pixel_ring.set_direction(direction)
                        logging.info('      Mosquito direction: {:.0f}°'.format(direction) + ' (processing time: {:.2f} s)'.format(processing_time))
                    else:
                        pixel_ring.off()
                        logging.info('      Mosquito direction: unknown')

    except KeyboardInterrupt:
        pass

    pixel_ring.off()


if __name__ == '__main__':
    main()
