# Co-Speech Gesture Generation: MambaTalk & GestureLSM Ablation Study

Project **Cassiopée** (Télécom SudParis, 2026), by **Alexandre Gonzalez** and **Mark Kriov**.

We study how robust two state-of-the-art speech-driven 3D gesture generators, **MambaTalk** and **GestureLSM**, are when training data is scarce (100 %, 50 % and 25 % of the training set), and we use a temporal **Grad-CAM** to see which parts of the audio the models rely on.

**Main finding:** MambaTalk stops using the audio signal when trained with 25 % of the data, even though its usual metrics stay stable. The full analysis is in the report: [`Report_Cassiopee2026_GONZALEZ_KIROV.pdf`](Report_Cassiopee2026_GONZALEZ_KIROV.pdf).

## Contributions

| | Work |
|---|---|
| **Alexandre Gonzalez** | MambaTalk side: data-ablation script (`Mambatalk/make_splits.py`), training and evaluation runs at 100 / 50 / 25 %, loss plots (`Mambatalk/plots/`), Grad-CAM analysis of MambaTalk, dataset rearrangement code, TextGrid generation for the demo (`GestureLSM/generate_textgrid.py`) |
| **Mark Kriov** | GestureLSM side: training and ablation runs, interpretability features added to the trainer and visualisation (`GestureLSM/trainer/`, `GestureLSM/visualize.py`), live demo (`GestureLSM/run_live_demo.py`) |

<!-- TODO : confirmer cette répartition avec Mark avant publication. -->

## Third-party code, data and licences

This repository bundles the original implementations of two published models, which we modified for our experiments. Their licences apply to their code:

- `Mambatalk/` : MambaTalk, with its own [`LICENSE`](Mambatalk/LICENSE)
- `GestureLSM/` : GestureLSM, released under Apache 2.0 according to its own README

See `Mambatalk/README.md` and `GestureLSM/README.md` for the original papers and citations. The experiments use the **BEAT2** dataset, which is not redistributed here: download it from its official source and follow its terms of use. Pretrained weights are not included either; get them by following the upstream READMEs.

---

## Repository Structure

* `Mambatalk/` : MambaTalk code with our ablation scripts, loss plots and Grad-CAM.
* `GestureLSM/` : GestureLSM code with our interpretability features and demo scripts.
* `Report_Cassiopee2026_GONZALEZ_KIROV.pdf` : the project report.

---

## 1. Dataset Processing & Ablation

We use the **BEAT2 (Body-Expression-Audio-Text) dataset**. Make sure it is downloaded and extracted, containing the `wave16k`, `smplxflame_30` and `textgrid` folders.

### Standard Setup
Update the dataset path in your configuration file (e.g. `configs/mambatalk.yaml`):
`data_path: /path/to/BEAT2/beat_english_v2.0.0/`

### Performing Data Ablation (50 % or 25 %)

#### MambaTalk
To test the models' robustness, we created a script that uniformly reduces the training data for specific speakers while keeping the exact same validation and test sets. It is called `make_splits.py`: run it to obtain the 50 % and 25 % datasets.

After swapping your `train_test_split.csv` with an ablated version, force the dataloader to rebuild the LMDB cache by setting `new_cache: True` in your `.yaml` config file.

#### GestureLSM
To reduce the training data uniformly, modify the `cache_generation` function in `beat_sep_lower.py` to keep the percentage you want before running a training.
Example: for 50 % of the training data, uncomment the `if not is_test:` part and set `half_count = original_count // 2`.

Note: always set `new_cache: True` in your config file when training a model with a different percentage of training data.

## 2. Training the Models

Once the dataset is prepared (and the cache is set to be rebuilt if changed), you can train a model from scratch.

### MambaTalk

Launch the training with:
`python train.py --config configs/mambatalk.yaml`

Or with: `bash run_scripts/train.sh` (check the GPU and config name variables first).

The checkpoints (`.bin` files) are saved in the directory given by `out_path` in your config file.

### GestureLSM

`CUDA_VISIBLE_DEVICES=1 python train.py --config configs_new/shortcut_rvqvae_128.yaml`

Use whichever GPU is available. We used the shortcut method, but you can use another one.

The checkpoints are saved in the `output_dir` set in the config file.

## 3. Evaluation & Inference

### MambaTalk

#### Generating Motion Sequences
To generate `.npz` gesture files from a trained model, make sure `test_ckpt` in your `.yaml` points to the checkpoint you want, then run:
`python test.py --config configs/mambatalk.yaml`

Or: `bash run_scripts/test.sh` (you can choose your GPU, as for training).

#### Rendering Videos and Visualization
To visualise the `.npz` outputs alongside the original audio, use the render script:
`xvfb-run -a python render.py --npy_path "outputs/audio2pose/custom/YOUR_RUN/res_2_scott_0_1_1.npz" --wav_path "/path/to/wave16k/2_scott_0_1_1.wav" --save_dir "outputs/render"`

You can pick another speaker than `2_scott_0_1_1`, but we recommend one with many sequences: a speaker with few sequences will not give a convincing result with the 25 % dataset.

We also recommend Blender to visualise your model: you can add the audio in *Video Editing* and synchronise it with your model.

#### Interpretability via Grad-CAM
We applied a temporal Grad-CAM to MambaTalk's audio encoders (face and body) to see how much the model relies on the audio when the training data shrinks. The resulting heatmaps align audio waveforms, spectrograms, TextGrid words and the model's gradient magnitude; they are shown as Figures 9 and 10 of the report.

<!-- TODO : ajouter Mambatalk/GradCam/gradcam.py puis remplacer ce paragraphe par la commande d'utilisation. -->
The MambaTalk Grad-CAM script is not included in this repository yet; the results and method are described in the report.

### GestureLSM

#### Generating Motion Sequences and Grad-CAM
To generate motion sequences (`.npz`) and Grad-CAM plots, run the test command:
`python train.py --config configs_new/NAME_OF_CONFIG.yaml --ckpt PATH_TO_YOUR_CHECKPOINT.ckpt --mode test`

You can visualise the result in Blender.

## 4. Demo

You can run a demo with your own audio and observe how your model behaves:
- Record yourself with a phone or any device (`.mp4` or `.m4a`). Vary the volume of your voice during the recording: the variations in movement are easier to observe.
- Add your audio file and run `generate_textgrid.py`. It converts your file to `.wav` and generates a TextGrid file, which gives the model the "context" (the words) it needs for specific movements.
- Run `run_live_demo.py`: it generates an `.npz` file that you can load in Blender to visualise your model.
