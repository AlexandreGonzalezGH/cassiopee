import whisper_timestamped as whisper
import textgrid
import librosa
import soundfile as sf

# 1. First, Whisper needs a WAV file, so we convert the m4a just like in your script
m4a_file = "TestDemo4.m4a"
wav_file = "temp_recording.wav"

print("Converting audio...")
y, sr = librosa.load(m4a_file, sr=16000)
sf.write(wav_file, y, sr)
duration = len(y) / sr

# 2. Load the Whisper AI model (the "base" model is small and fast)
print("Loading Whisper AI to transcribe and align...")
audio = whisper.load_audio(wav_file)
model = whisper.load_model("base") # Use "small" or "medium" for better accuracy

# 3. Transcribe the audio and get exact word timestamps
# Change language="en" to "fr" if you are speaking French!
result = whisper.transcribe(model, audio, language="en")

# 4. Build the TextGrid file using the exact timestamps
print("Building TextGrid...")
tg = textgrid.TextGrid(maxTime=duration)
tier = textgrid.IntervalTier(name='words', maxTime=duration)

for segment in result["segments"]:
    for word in segment["words"]:
        start = word["start"]
        end = word["end"]
        text = word["text"].strip()
        
        # FIX 1: Ignore words that Whisper thinks start after the audio ends
        if start >= duration:
            continue
            
        # FIX 2: Clamp the end time so it doesn't exceed the audio length
        if end > duration:
            end = duration
            
        # FIX 3: Ultimate safety check to prevent ValueError
        if start >= end:
            continue
            
        tier.add(start, end, text)

tg.append(tier)


# 5. Save it!
textgrid_output = "my_live_recording.TextGrid"
tg.write(textgrid_output)

print(f"✅ Success! Generated {textgrid_output}")