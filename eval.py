# eval_model.py
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from classes.utils_audio import load_wav, save_wave, stft_np, magphase
from classes.model_class import SpeechDenoisingModel

# -----------------------
# Config loading
# -----------------------
CONFIG_PATH = Path("config.json")
with CONFIG_PATH.open("r") as f:
    CONFIG = json.load(f)

GLOBAL_CFG = CONFIG["global"]
PATHS_CFG = CONFIG["paths"]
EVAL_CFG = CONFIG["eval"]

SAMPLE_RATE = int(GLOBAL_CFG["sample_rate"])
N_FFT = int(GLOBAL_CFG["stft"]["n_fft"])
HOP_LENGTH = int(GLOBAL_CFG["stft"]["hop_length"])

VAL_AUDIO_DIR = Path(PATHS_CFG["processed_val_dir"])
OUTPUT_DIR = Path(PATHS_CFG["eval_output_dir"])

NUM_PASSES = int(EVAL_CFG["num_passes"])
SAMPLE_INDEX = int(EVAL_CFG.get("sample_index", 0))

PHASE_SUPPORT = EVAL_CFG.get("phase_support", True)

device_cfg = EVAL_CFG.get("device", "cpu")
if device_cfg == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = device_cfg

CHECKPOINT_PATH = Path(PATHS_CFG["checkpoint_path"])


def plot_spectrogram_stack(
    waveforms: list[np.ndarray],
    labels: list[str],
    out_path: Path,
) -> None:
    """
    Plot a vertical stack of spectrograms:
    one row per waveform, same STFT config for all.
    """
    assert len(waveforms) == len(labels)

    num_rows = len(waveforms)
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(10, 3 * num_rows),
        squeeze=False,
    )

    for i, (wave, label) in enumerate(zip(waveforms, labels)):
        spec = stft_np(wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag, _ = magphase(spec)
        mag_log = np.log1p(mag)

        ax = axes[i, 0]
        im = ax.imshow(mag_log, origin="lower", aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Frequency bins")

        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_model_architecture(model, out_path: Path, input_shape):  # Changed to 2 channels
    """
    Create a visual diagram of the model architecture using torchviz with hooks.
    """
    try:
        from torchviz import make_dot
        import subprocess
        
        # Check if Graphviz is available
        try:
            result = subprocess.run(['dot', '-V'], check=True, capture_output=True, text=True)
            print(f"Graphviz found: {result.stderr.strip()}")
        except FileNotFoundError:
            print("Graphviz 'dot' command not found in PATH.")
            print("Install Graphviz:")
            print("  macOS: brew install graphviz")
            print("  Ubuntu/Debian: sudo apt-get install graphviz")
            print("  Windows: Download from https://graphviz.org/download/")
            return
        except subprocess.CalledProcessError as e:
            print(f"Graphviz error: {e}")
            return

        print("Creating detailed model architecture diagram with layer-by-layer visualization...")
        
        # Create dummy input with requires_grad=True - now with 2 channels for magnitude + phase
        dummy_input = torch.randn(input_shape, requires_grad=True).to(model.device)
        
        # Dictionary to store intermediate outputs
        layer_outputs = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.requires_grad:
                    layer_outputs[name] = output
            return hook
        
        # Register hooks for all modules
        for name, module in model.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # Set model to training mode and do forward pass
        model.model.train()
        output = model.model(dummy_input)
        
        # Add final output to layer_outputs
        layer_outputs['final_output'] = output
        
        # Create the main computation graph
        params_dict = dict(model.model.named_parameters())
        params_dict['input'] = dummy_input
        
        # Create comprehensive graph
        dot = make_dot(
            output,
            params=params_dict,
            show_attrs=True,
            show_saved=True
        )
        
        # Customize graph appearance for high quality
        dot.graph_attr.update({
            'rankdir': 'TB',
            'size': '20,30',
            'dpi': '300',
            'bgcolor': 'white',
            'fontsize': '14',
            'fontname': 'Arial',
            'resolution': '300'
        })
        
        dot.node_attr.update({
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'lightblue',
            'fontsize': '12',
            'fontname': 'Arial',
            'margin': '0.3',
            'width': '2',
            'height': '0.8'
        })
        
        dot.edge_attr.update({
            'fontsize': '10',
            'fontname': 'Arial'
        })
        
        # Save high-resolution PNG
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dot.format = "png"
        dot.render(str(out_path.with_suffix("")), cleanup=True)
        print(f"High-resolution PNG saved: {out_path}")
        
        # Save SVG version
        svg_path = out_path.with_suffix('.svg')
        dot.format = "svg"
        dot.render(str(svg_path.with_suffix("")), cleanup=True)
        print(f"SVG version saved: {svg_path}")

        # Clean up hooks
        for hook in hooks:
            hook.remove()
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")
    except Exception as e:
        print(f"Error creating model diagram: {e}")
        import traceback
        traceback.print_exc()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    noisy_val_files = sorted(VAL_AUDIO_DIR.glob("noisy_*.wav"))
    if not noisy_val_files:
        raise RuntimeError(f"No noisy val files found in {VAL_AUDIO_DIR}")

    # choose index within range
    idx = min(max(0, SAMPLE_INDEX), len(noisy_val_files) - 1)
    noisy_path = noisy_val_files[idx]

    print(f"Using val noisy sample index {idx}: {noisy_path}")

    # load noisy input
    noisy_wave = load_wav(noisy_path, SAMPLE_RATE)

    # try to load matching clean ground truth if it exists
    clean_path = VAL_AUDIO_DIR / f"clean_{idx:05d}.wav"
    clean_wave = None
    if clean_path.exists():
        clean_wave = load_wav(clean_path, SAMPLE_RATE)
        print(f"Found ground truth clean file: {clean_path}")
    else:
        print("No matching clean_XXXXX.wav found; proceeding without ground truth.")

    # init model
    model = SpeechDenoisingModel(
        device=DEVICE,
        learning_rate=1e-3,
        checkpoint_path=str(CHECKPOINT_PATH),
        phase_support=PHASE_SUPPORT
    )
    model.load_checkpoint()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # create model architecture diagram
    # model_diagram_path = OUTPUT_DIR / "model_architecture.png"
    # plot_model_architecture(model, model_diagram_path, input_shape=(1, 2 if PHASE_SUPPORT else 1, 257, 256))

    # run multipass denoising
    waveforms_for_plot = []
    labels_for_plot = []

    # add ground truth first if available
    if clean_wave is not None:
        waveforms_for_plot.append(clean_wave)
        labels_for_plot.append("Clean (ground truth)")

    # original noisy input
    waveforms_for_plot.append(noisy_wave)
    labels_for_plot.append("Noisy input")

    current = noisy_wave
    for p in range(NUM_PASSES):
        print(f"Running pass {p+1}/{NUM_PASSES}")
        denoised, _, _ = model.infer(current, N_FFT, HOP_LENGTH)
        current = denoised

        # save audio for this pass
        out_wav = OUTPUT_DIR / f"denoised_sample{idx:03d}_pass{p+1}.wav"
        save_wave(current, out_wav, SAMPLE_RATE)
        print(f"Saved: {out_wav}")

        # store for spectrogram visualization
        waveforms_for_plot.append(current)
        labels_for_plot.append(f"Denoised (pass {p+1})")

    # create stacked spectrogram figure
    spec_path = OUTPUT_DIR / f"spectrogram_sample{idx:03d}.png"
    plot_spectrogram_stack(waveforms_for_plot, labels_for_plot, spec_path)
    print(f"Saved spectrogram stack: {spec_path}")

    print("Done.")


if __name__ == "__main__":
    main()