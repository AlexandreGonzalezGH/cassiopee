# Co-Speech Gesture Generation: MambaTalk & GestureLSM Ablation Study

This repository contains the source code, dataset processing scripts, evaluation tools, and final reports (PDF and Poster) for our project on speech-driven 3D gesture generation. 

The primary focus of this work is investigating the robustness of state-of-the-art gesture synthesis models (**MambaTalk** and **GestureLSM**) under data-scarce conditions (100%, 50%, and 25% training data ablation). We also provide a custom **Grad-CAM** implementation to interpret the audio-to-motion attention maps.

---

## Repository Structure

* `configs/` : YAML configuration files for the models.
* `dataloaders/` : Scripts for loading and caching the BEAT2 dataset.
* `GradCam/` : Custom implementation of temporal Grad-CAM for MambaTalk/GestureLSM.
* `outputs/` : Generated `.npz` motions, rendered `.mp4` videos, and Grad-CAM heatmaps.
* `pretrained/` : Pre-trained Vector Quantized Auto-Encoders (VQ-VAE) and baseline models.
* `92_Poster/` : Contains the project poster.

---

## 1. Dataset Processing & Ablation

We utilize the **BEAT2 (Body-Expression-Audio-Text) dataset**. Ensure the dataset is downloaded and extracted, containing the `wave16k`, `smplxflame_30`, and `textgrid` folders.

### Standard Setup
Update the dataset path in your configuration file (e.g., "configs/mambatalk.yaml"):
`data_path: /path/to/BEAT2/beat_english_v2.0.0/`

### Performing Data Ablation (50% or 25%)
To test the models' robustness, we created a script to uniformly reduce the training data for specific speakers while preserving the exact same validation and test sets. It is called "make_splits.py". You just have to run it to obtain the 50% and 25% dataset.

After swapping your "train_test_split.csv" with an ablated version, you must force the dataloader to rebuild the LMDB cache. In your .yaml config file, set : `new_cache: True`

## 2. Training the Models

Once the dataset is prepared (and the cache is set to be rebuilt if changed), you can train the model from scratch.
### MambaTalk

Launch the training pipeline with this command :
`python train.py --config configs/mambatalk.yaml`

Or just with : `bash run_scripts/train.sh`
Note : For the second option, you have to check the gpu and configname variable.

The checkpoints (.bin files) will be saved in the directory specified by out_path in your config file (i.e. the mambatalk.yaml file).

## 3. Evaluation & Inference

### Mambatalk

#### Generating Motion Sequences
To generate .npz gesture files from a trained model, ensure test_ckpt in your .yaml points to your desired model checkpoint. Then run :
`python test.py --config configs/mambatalk.yaml`

Or just : `bash run_scripts/test.sh` in which you can choose your gpu, just as for the training.

#### Rendering Videos and Visualization
To visualize the .npz outputs alongside the original audio, use the render script :
`xvfb-run -a python render.py --npy_path "outputs/audio2pose/custom/YOUR_RUN/res_2_scott_0_1_1.npz" --wav_path "/path/to/wave16k/2_scott_0_1_1.wav" --save_dir "outputs/render"`

Note : you can choose any other speaker besides 2_scott_0_1_1 of course. But we recommend to use a speaker who has many sequences. It will be hard to have a performative speaker if he has few sequences for the 25% datasets (not enough data).

BUT we recommend to you to use Blender to visualize your model. You can also put the audio in "Video Editing->Video Editing" and synchronize it with your model.

#### Interpretability via Grad-CAM
To analyze how the model relies on acoustic cues (and how data ablation smooths out this attention), use our custom Grad-CAM implementation:
`python GradCam/gradcam.py`

This script will output both 1D and 2D temporal heatmaps in the outputs/gradcam/ folder, aligning audio waveforms, spectrograms, TextGrid words, and the model's gradient magnitude.

## 4. Demo
You can make a Demo with your own audio and observe how your model behaves : 
- First, record yourselves with your phone or any other device to have a .mp4 or .m4a file. We suggest you to modify your volume voice during the recording so you will observe the variation in movements much more easier.
- Then, you can add your audio file and run the "generate_textgrid.py" script. It will first convert your file into a .wav file. After that, it will generate a TextGrid file which is important to give the "context" to your model. Indeed, the model isn't able to do specific movements if it doesn't know the specific word associated to this movement.
- Finally, you can run the "run_live_demo.py" file. It will generate a .npz file that you can put in Blender to visualize your model. 

