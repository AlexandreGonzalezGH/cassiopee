import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(npy_path, wav_path, output_html_path):
    print(f"Loading {npy_path} and {wav_path}...")
    
    # Load the raw attention math you extracted
    frame_importance = np.load(npy_path)
    
    # Load the audio
    audio, sr = librosa.load(wav_path, sr=16000)
    
    fps = 30
    duration = len(frame_importance) / fps
    audio = audio[:int(duration * sr)]
    
    # Create time axes
    time_audio = np.linspace(0, duration, len(audio))
    time_frames = np.linspace(0, duration, len(frame_importance))

    print("Building interactive dashboard...")
    # Create a 2-row subplot, linking the X-axis so zooming one zooms both!
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.3, 0.7])

    # Top Graph: The Attention Heatmap
    fig.add_trace(
        go.Scatter(
            x=time_frames, 
            y=frame_importance,
            fill='tozeroy',
            mode='lines',
            line=dict(color='cyan', width=2),
            fillcolor='rgba(0, 255, 255, 0.2)',
            name='AI Attention',
            hovertemplate='Time: %{x:.2f}s<br>Attention: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Bottom Graph: The Raw Audio Waveform
    step = max(1, len(audio) // 10000) 
    fig.add_trace(
        go.Scatter(
            x=time_audio[::step], 
            y=audio[::step],
            mode='lines',
            line=dict(color='magenta', width=1),
            name='Audio Waveform',
            hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Make it look like a slick, modern UI
    fig.update_layout(
        template="plotly_dark",
        title="GestureLSM Interactive Saliency Analysis",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified", 
        showlegend=False
    )

    fig.update_yaxes(title_text="Attention Weight", row=1, col=1)
    fig.update_yaxes(title_text="Audio Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

    # Save as a standalone HTML file
    fig.write_html(output_html_path)
    print(f"Done! Open {output_html_path} in any web browser.")

# Run it!
create_interactive_dashboard(
    npy_path="./outputs/weights_25pct/shortcut_rvqvae_128/999/gradcam_2_scott_0_1_1.npy", 
    wav_path="./datasets/BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_1_1.wav", 
    output_html_path="final_dashboard.html"
)