import sounddevice as sd
import numpy as np
import scipy.signal as signal
import speech_recognition as sr
import tempfile
import scipy.io.wavfile as wav

# ================= CONFIG =================
SAMPLE_RATE = 16000       # 16 kHz (Nyquist-safe for speech)
DURATION = 10              # seconds
ENERGY_THRESHOLD = 0.0002  # signal energy threshold
NOISE_FRAMES = 10      # frames used to estimate noise
ALPHA = 4.0           # adaptive threshold multiplier
MIN_SNR_DB = 10       # minimum acceptable SNR
MIN_NOISE_ENERGY = 1e-6


def float_to_pcm16(audio):
    audio = audio / np.max(np.abs(audio))  # normalize
    audio_int16 = np.int16(audio * 32767)
    return audio_int16

def estimate_noise_energy(stream, frame_size):
    noise_energies = []
    for _ in range(NOISE_FRAMES):
        frame, _ = stream.read(frame_size)
        frame = frame.flatten()
        energy = np.mean(frame ** 2)
        noise_energies.append(energy)
    avg_noise = np.mean(noise_energies)
    return max(avg_noise, MIN_NOISE_ENERGY)

def compute_snr(signal_energy, noise_energy):
    if noise_energy == 0:
        return float('inf')
    return 10 * np.log10(signal_energy / noise_energy)

# ================= RECORD AUDIO =================
def record_audio_vad():
    print("\n🎤 Speak now...")

    frame_duration = 0.03  # 30 ms
    frame_size = int(SAMPLE_RATE * frame_duration)
    silence_limit = 1.0
    silence_frames = int(silence_limit / frame_duration)

    recorded_frames = []
    silent_count = 0
    speech_detected = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    ) as stream:

        # ---- Noise estimation (adaptive threshold) ----
        noise_energy = estimate_noise_energy(stream, frame_size)
        adaptive_threshold = ALPHA * noise_energy

        print(f" Noise Energy: {noise_energy:.6f}")
        print(f" Adaptive Threshold: {adaptive_threshold:.6f}")

        while True:
            frame, _ = stream.read(frame_size)
            frame = frame.flatten()
            energy = np.mean(frame ** 2)

            if energy > adaptive_threshold:
                speech_detected = True
                silent_count = 0
                recorded_frames.append(frame)
            elif speech_detected:
                silent_count += 1
                recorded_frames.append(frame)

            if speech_detected and silent_count > silence_frames:
                break

 #   print("✔ Voice activity ended")
    return np.concatenate(recorded_frames), noise_energy

# ================= DSP PREPROCESSING =================
def preprocess_signal(audio):
    # High-pass filter to remove low-frequency noise
    b, a = signal.butter(4, 100 / (SAMPLE_RATE / 2), btype='high')
    filtered = signal.filtfilt(b, a, audio)
    return filtered

# ================= SIGNAL QUALITY CHECK =================
def signal_quality_check(audio, noise_energy):
    signal_energy = np.mean(audio ** 2)
    rms = np.sqrt(signal_energy)
    snr_db = compute_snr(signal_energy, noise_energy)

    print(f"Signal Energy: {signal_energy:.6f}")
    print(f"RMS Amplitude: {rms:.6f}")
    print(f"Estimated SNR: {snr_db:.2f} dB")

    if snr_db < MIN_SNR_DB:
        print("SNR below threshold (10 dB). Voice rejected.")
        return False

    print("Signal accepted (SNR ≥ 10 dB).")
    return True


# ================= SPEECH TO TEXT =================
def speech_to_text(audio):
    recognizer = sr.Recognizer()

    pcm_audio = float_to_pcm16(audio)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, SAMPLE_RATE, pcm_audio)
        with sr.AudioFile(f.name) as source:
            audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# ================= MAIN PIPELINE =================
def get_voice_query():
    audio, noise_energy = record_audio_vad()

    filtered_audio = preprocess_signal(audio)

    if not signal_quality_check(filtered_audio, noise_energy):
        return None


    print("Signal accepted. Performing speech recognition...")    
    text = speech_to_text(filtered_audio)

    if text is None:
        print("❌ Speech could not be recognized.")
        return None

    print(f"Recognized Text: {text}")
    return text


# ================= TEST =================
if __name__ == "__main__":
    query = get_voice_query()
    if query:
        print("\n➡ Pass this text to chatbot:")
        print(query)
