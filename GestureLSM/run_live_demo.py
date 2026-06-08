import os
import subprocess
import librosa
import soundfile as sf
import textgrid
import numpy as np

# --- CONFIGURATIONS ---
# Put the name of the file you record on your phone/mic right here!
RAW_AUDIO_INPUT = "TestDemo.m4a"  

# Don't change these unless your paths changed
FORMATTED_NAME = "2_custom_0_1_1"
SANDBOX_DIR = "./custom_dataset"
CKPT_PATH = "./outputs/weights_100pct/shortcut_rvqvae_128/best_fgd/ckpt.pth"
# ----------------------

def run_pipeline():
    print("\n" + "="*50)
    print("🚀 INITIATING LIVE DEMO PIPELINE")
    print("="*50)

    # Make sure sandbox directories exist
    os.makedirs(os.path.join(SANDBOX_DIR, "wave16k"), exist_ok=True)
    os.makedirs(os.path.join(SANDBOX_DIR, "smplxflame_30"), exist_ok=True)
    os.makedirs(os.path.join(SANDBOX_DIR, "textgrid"), exist_ok=True)

    print("\n[1/3] Processing and Formatting Audio...")
    if not os.path.exists(RAW_AUDIO_INPUT):
        print(f"❌ ERROR: Could not find {RAW_AUDIO_INPUT}! Please drop it in the folder.")
        return

    y, sr = librosa.load(RAW_AUDIO_INPUT, sr=16000, mono=True)
    duration = len(y) / sr
    frames = int(duration * 30)

    wav_path = os.path.join(SANDBOX_DIR, f"wave16k/{FORMATTED_NAME}.wav")
    sf.write(wav_path, y, sr)
    print(f"✅ Audio converted to 16kHz Mono: {duration:.2f} seconds.")

    print("\n[2/3] Building Dataset Sandbox...")
    # Generate Dummy Poses
    pose_dest = os.path.join(SANDBOX_DIR, f"smplxflame_30/{FORMATTED_NAME}.npz")
    np.savez(pose_dest, poses=np.zeros((frames, 165)), trans=np.zeros((frames, 3)), 
             expressions=np.zeros((frames, 100)), betas=np.zeros(300))
    
    # Generate Perfect TextGrid
    tg_path = os.path.join(SANDBOX_DIR, f"textgrid/{FORMATTED_NAME}.TextGrid")
    tg = textgrid.TextGrid(maxTime=duration)
    tier = textgrid.IntervalTier(name='words', maxTime=duration)
    tier.add(0.0, duration, "hello") # Dummy word for the tokenizer
    tg.append(tier)
    tg.write(tg_path)
    print("✅ Sandbox injected with dummy variables.")

    print("\n[3/3] Executing Neural Network Inference...")
    test_command = [
        "python", "train.py", 
        "--config", "configs_new/shortcut_custom_dataset.yaml", 
        "--ckpt", CKPT_PATH, 
        "--mode", "test"
    ]
    
    # Run the model!
    subprocess.run(test_command)

    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETE!")
    print(f"Check your outputs folder for res_{FORMATTED_NAME}.npz and the GradCAM heatmap!")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()