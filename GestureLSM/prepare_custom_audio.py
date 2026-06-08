import numpy as np
import librosa
import os
import shutil

def hack_dataset_for_custom_audio(wav_path, dataset_dir="./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/"):
    # The dataloader uses the filename to extract IDs, so we keep the BEAT naming convention
    file_name = os.path.basename(wav_path).replace('.wav', '')
    
    print(f"Loading custom audio: {file_name}")
    audio, sr = librosa.load(wav_path, sr=16000)
    duration = len(audio) / 16000
    fps = 30
    frames = int(duration * fps)
    
    # 1. Copy the Audio to the wave16k folder
    audio_dest = os.path.join(dataset_dir, f"wave16k/{file_name}.wav")
    os.makedirs(os.path.dirname(audio_dest), exist_ok=True)
    shutil.copy(wav_path, audio_dest)
    print(f"Copied audio to {audio_dest}")

    # 2. Create a Dummy .npz Pose File
    # The dataloader reads this just to get the 'length' of the sequence. 
    # The AI will completely overwrite these zeros with its own generated motion.
    pose_dest = os.path.join(dataset_dir, f"smplxflame_30/{file_name}.npz")
    os.makedirs(os.path.dirname(pose_dest), exist_ok=True)
    np.savez(pose_dest,
             poses=np.zeros((frames, 165)),   # Standard SMPL-X pose dimension
             trans=np.zeros((frames, 3)),     # Translation
             expressions=np.zeros((frames, 100)), # Facial expressions
             betas=np.zeros(300))             # Body shape
    print(f"Generated dummy pose file: {pose_dest}")

    # 3. Create a Dummy .TextGrid File
    # The model expects transcripts. We feed it an empty string so it relies 100% on the audio.
    tg_dest = os.path.join(dataset_dir, f"textgrid/{file_name}.TextGrid")
    os.makedirs(os.path.dirname(tg_dest), exist_ok=True)
    tg_content = f"""File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = {duration}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = {duration}
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = {duration}
            text = ""
"""
    with open(tg_dest, "w") as f:
        f.write(tg_content)
    print(f"Generated dummy TextGrid: {tg_dest}")
    
    print("\nSUCCESS! Your files are in the dataset.")
    print("--------------------------------------------------")
    print(f"ACTION REQUIRED: Open your train_test_split.csv file and add this exact line to the bottom:")
    print(f"{file_name},test")
    print("--------------------------------------------------")

# Run the hacker script! 
# (Make sure to put your actual audio file path here)
hack_dataset_for_custom_audio("2_custom_0_1_1.wav")