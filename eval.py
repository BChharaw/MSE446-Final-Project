# eval_model.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from classes.model_class import SpeechDenoisingModel
from classes.utils_audio import load_wav, magphase, save_wave, stft_np

# ============================================================================
# Configuration loading and global evaluation setup
# ============================================================================
# This script is the evaluation / visualization counterpart to train.py.
# It is designed to:
#   1) Load a single noisy validation clip (optionally with its clean target).
#   2) Restore a trained SpeechDenoisingModel from a checkpoint.
#   3) Run *multi-pass* inference: feed the model's output back as input
#      several times to study how the denoising behaves iteratively.
#   4) Save per-pass audio outputs and a stacked spectrogram figure that lets
#      you compare:
#           - Ground truth clean (if available),
#           - Original noisy input,
#           - Denoised outputs for each pass.
#
# Everything is driven by config.json to keep evaluation fully aligned with
# training (sample rate, STFT parameters, model type, checkpoint path, etc.).
# ============================================================================
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

MODEL_TYPE = EVAL_CFG.get("model_type", "small")
NUM_PASSES = int(EVAL_CFG["num_passes"])
SAMPLE_INDEX = int(EVAL_CFG.get("sample_index", 0))

PHASE_SUPPORT = EVAL_CFG.get("phase_support", True)

device_cfg = EVAL_CFG.get("device", "cpu")
if device_cfg == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = device_cfg

CHECKPOINT_PATH = Path(EVAL_CFG["checkpoint_path"])


def plot_spectrogram_stack(
    waveforms: list[np.ndarray],
    labels: list[str],
    out_path: Path,
) -> None:
    """
    Plot a vertical stack of spectrograms:
    one row per waveform, same STFT config for all.

    This function is meant for *qualitative* inspection:
      - It runs the same STFT configuration used during training (N_FFT, HOP_LENGTH)
        on each waveform.
      - Magnitudes are converted to log scale to emphasize low-energy details and
        make differences between noisy and denoised signals visually obvious.
      - Each row gets its own colorbar, so dynamic range differences between
        passes are visible rather than being flattened by a global normalization.

    The resulting PNG is a compact visual summary of how the model changes the
    time-frequency structure from noisy input to successive denoised outputs (and
    clean, if available).
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


def plot_model_architecture(model, out_path: Path, input_shape):
    """
    Create a simplified visual diagram showing only major convolutional layers
    and connections using Graphviz.

    Intent:
      - This is *not* a perfect structural representation of every detail in
        the PyTorch module graph. Instead, it extracts a human-readable summary
        of the main Conv / ConvTranspose / Pool / Upsample / RNN blocks and
        annotates skip connections when they can be inferred from naming.
      - It is meant to be included in a report/paper as a high-level schematic,
        keeping the diagram simple and uncluttered while still reflecting the
        encoder–bottleneck–decoder structure.

    Requirements:
      - Python package: `graphviz`
      - System Graphviz installation (e.g., `brew install graphviz` on macOS)
    """
    try:
        from graphviz import Digraph

        print("Creating simplified model architecture diagram...")

        # Create a new directed graph
        dot = Digraph(comment="Model Architecture", format="png")
        dot.attr(rankdir="LR", size="5,6", dpi="300", ranksep="0.2")
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontname="Arial",
            width="1.2",
            height="0.6",
        )
        dot.attr("edge", fontname="Arial", fontsize="5")

        # Iterate through model and extract conv/pooling/upsampling layers.
        # We rely on naming conventions inside model.model.named_modules()
        # to infer encoder levels and decoder/upsampling blocks, which allows
        # us to add skip connections between matching levels when present.
        layer_count = 0
        prev_name = "input"
        skip_connections = {}  # Track skip connections for UNet {level: last_node_name}
        encoder_outputs = {}  # Track encoder outputs by level
        current_level = 0

        for name, module in model.model.named_modules():
            # Detect encoder/decoder level from module name
            if "enc1" in name and "enc" in name:
                current_level = 1
            elif "enc2" in name:
                current_level = 2
            elif "bottleneck" in name:
                current_level = 0
            elif "up2" in name or "dec2" in name:
                current_level = 2
            elif "up1" in name or "dec1" in name:
                current_level = 1

            if isinstance(module, torch.nn.Conv2d):
                layer_count += 1
                node_name = f"conv{layer_count}"
                label = f"Conv2D\n{module.in_channels}→{module.out_channels}\nkernel={module.kernel_size[0]}"
                dot.node(node_name, label, fillcolor="lightblue")
                dot.edge(prev_name, node_name)

                # Track last node of each encoder level for potential skip links
                if "enc" in name and current_level > 0:
                    encoder_outputs[current_level] = node_name

                prev_name = node_name

            elif isinstance(module, torch.nn.ConvTranspose2d):
                layer_count += 1
                node_name = f"convT{layer_count}"
                label = f"ConvTranspose2D\n{module.in_channels}→{module.out_channels}\nkernel={module.kernel_size[0]}"
                dot.node(node_name, label, fillcolor="lightyellow")
                dot.edge(prev_name, node_name)

                # Add skip connection from corresponding encoder level when found.
                if current_level > 0 and current_level in encoder_outputs:
                    dot.edge(
                        encoder_outputs[current_level],
                        node_name,
                        style="dashed",
                        color="red",
                        label=f"skip{current_level}",
                        constraint="false",
                    )

                prev_name = node_name

            elif isinstance(module, torch.nn.MaxPool2d):
                layer_count += 1
                node_name = f"pool{layer_count}"
                label = f"MaxPool2D\nkernel={module.kernel_size}"
                dot.node(node_name, label, fillcolor="lightcoral")
                dot.edge(prev_name, node_name)
                prev_name = node_name

            elif isinstance(module, torch.nn.Upsample):
                layer_count += 1
                node_name = f"upsample{layer_count}"
                label = f"Upsample\nscale={module.scale_factor}"
                dot.node(node_name, label, fillcolor="lightcoral")
                dot.edge(prev_name, node_name)
                prev_name = node_name

            elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer_count += 1
                node_name = f"rnn{layer_count}"
                rnn_type = module.__class__.__name__
                label = f"{rnn_type}\ninput={module.input_size}\nhidden={module.hidden_size}\nlayers={module.num_layers}"
                dot.node(node_name, label, fillcolor="lightpink")
                dot.edge(prev_name, node_name)
                prev_name = node_name

        # Add output node at the end of the chain.
        dot.node(
            "output",
            "Output\n2 channels" if PHASE_SUPPORT else "Output\n1 channel",
            fillcolor="lightgreen",
        )
        dot.edge(prev_name, "output")

        # Save the diagram in multiple formats for flexibility in reports.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dot.render(str(out_path.with_suffix("")), cleanup=True)
        print(f"Simplified architecture diagram saved: {out_path}")

        # Save high-resolution PNG
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dot.format = "png"
        dot.render(str(out_path.with_suffix("")), cleanup=True)
        print(f"High-resolution PNG saved: {out_path}")

        # Also save SVG version (vector graphic for publication-quality figures)
        dot.format = "svg"
        svg_path = out_path.with_suffix(".svg")
        dot.render(str(svg_path.with_suffix("")), cleanup=True)
        print(f"SVG version saved: {svg_path}")

    except ImportError:
        print("graphviz not installed. Install with: pip install graphviz")
        print(
            "Also ensure Graphviz system package is installed (brew install graphviz on macOS)"
        )
    except Exception as e:
        print(f"Error creating model diagram: {e}")
        import traceback

        traceback.print_exc()


def main():
    # Ensure the evaluation output directory exists before writing any files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate validation noisy files and select one by index. This keeps the
    # evaluation deterministic: the same SAMPLE_INDEX in the config always maps
    # to the same (sorted) file in VAL_AUDIO_DIR.
    noisy_val_files = sorted(VAL_AUDIO_DIR.glob("noisy_*.wav"))
    if not noisy_val_files:
        raise RuntimeError(f"No noisy val files found in {VAL_AUDIO_DIR}")

    # Clamp index into valid range to avoid out-of-bounds errors.
    idx = min(max(0, SAMPLE_INDEX), len(noisy_val_files) - 1)
    noisy_path = noisy_val_files[idx]

    print(f"Using val noisy sample index {idx}: {noisy_path}")

    # Load noisy input waveform using the same sample rate as training.
    noisy_wave = load_wav(noisy_path, SAMPLE_RATE)

    # Try to load matching clean ground truth if it exists. This is purely for
    # visualization and listening; inference itself does not depend on having
    # clean targets at evaluation time.
    clean_path = VAL_AUDIO_DIR / f"clean_{idx:05d}.wav"
    clean_wave = None
    if clean_path.exists():
        clean_wave = load_wav(clean_path, SAMPLE_RATE)
        print(f"Found ground truth clean file: {clean_path}")
    else:
        print("No matching clean_XXXXX.wav found; proceeding without ground truth.")

    # Initialize model and load checkpoint. Learning rate here is unused for
    # inference (no optimization occurs), but the SpeechDenoisingModel API
    # expects it; keeping the value simple avoids accidental side effects.
    model = SpeechDenoisingModel(
        device=DEVICE,
        learning_rate=1e-3,
        model_type=MODEL_TYPE,
        checkpoint_path=str(CHECKPOINT_PATH),
        phase_support=PHASE_SUPPORT,
    )
    model.load_checkpoint()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Optionally create a model architecture diagram for documentation.
    if EVAL_CFG.get("vis_model", True):
        model_diagram_path = OUTPUT_DIR / "model_architecture.png"
        plot_model_architecture(
            model,
            model_diagram_path,
            input_shape=(1, 2 if PHASE_SUPPORT else 1, 257, 256),
        )

    # Prepare data structures for spectrogram plotting.
    waveforms_for_plot = []
    labels_for_plot = []

    # Add ground truth first if available so it appears at the top of the figure.
    if clean_wave is not None:
        waveforms_for_plot.append(clean_wave)
        labels_for_plot.append("Clean (ground truth)")

    # Always include the original noisy input for comparison.
    waveforms_for_plot.append(noisy_wave)
    labels_for_plot.append("Noisy input")

    # Multi-pass denoising:
    #   - The first pass runs on the original noisy input.
    #   - Each subsequent pass runs on the previous denoised output, effectively
    #     letting you inspect whether multiple iterations oversmooth, introduce
    #     artifacts, or further clean residual noise.
    current = noisy_wave
    for p in range(NUM_PASSES):
        print(f"Running pass {p+1}/{NUM_PASSES}")
        denoised, _, _ = model.infer(current, N_FFT, HOP_LENGTH)
        current = denoised

        # Save audio for this pass so you can listen to progressive improvements.
        out_wav = OUTPUT_DIR / f"denoised_sample{idx:03d}_pass{p+1}.wav"
        save_wave(current, out_wav, SAMPLE_RATE)
        print(f"Saved: {out_wav}")

        # Store for spectrogram visualization stack.
        waveforms_for_plot.append(current)
        labels_for_plot.append(f"Denoised (pass {p+1})")

    # Create stacked spectrogram figure summarizing the whole pipeline.
    spec_path = OUTPUT_DIR / f"spectrogram_sample{idx:03d}.png"
    plot_spectrogram_stack(waveforms_for_plot, labels_for_plot, spec_path)
    print(f"Saved spectrogram stack: {spec_path}")

    print("Done.")


if __name__ == "__main__":
    main()
