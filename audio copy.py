import pyaudio
import numpy as np
from scipy.signal import butter, lfilter

# --- Audio settings ---
CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- Band-pass settings ---
FREQ_LOW = 750
FREQ_HIGH = 850

# --- Filter Design ---
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(FREQ_LOW, FREQ_HIGH, RATE)

def apply_bandpass_filter(data, b, a):
    return lfilter(b, a, data)

# --- PyAudio Setup ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("🎧 Listening for mosquito wingbeat (650–850 Hz)... Ctrl+C to stop.")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)

        # Apply band-pass filter
        filtered = apply_bandpass_filter(samples, b, a)

        # Compute signal energy
        energy = np.sqrt(np.mean(filtered**2))  # RMS energy

        if energy > 100:  # Threshold — tweak this for your setup
            print("🦟 Mosquito detected! Energy:", int(energy))
        else:
            print("...")

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()