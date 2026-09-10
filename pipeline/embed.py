"""Turn downloaded audio into CLAP embeddings plus zero-shot tags.

CLAP maps audio and text into the same 512-dimensional space, which buys
two things at once: a vector per track for the map, and free labelling by
comparing each track against text prompts.

Reads data/manifest.json, writes data/embeddings.npz.
Safe to re-run: tracks already embedded are skipped.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, ClapModel

warnings.filterwarnings("ignore", category=UserWarning)

# The music_and_speech variant, not plain music: this corpus contains narrated
# discs and vocal sides, and the speech-aware model handles them less badly.
MODEL_NAME = "laion/larger_clap_music_and_speech"

SAMPLE_RATE = 48000
WINDOW_SECONDS = 10
MIN_WINDOW_SECONDS = 1  # trailing fragments shorter than this are dropped

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"

# Prompts for zero-shot tagging. Grouped so each group can be scored as its own
# softmax: asking "which instrument" and "which mood" separately beats throwing
# every label into one pile, because the groups are not mutually exclusive.
LABEL_GROUPS = {
    "ensemble": [
        "a solo piano piece",
        "a solo violin",
        "a string quartet",
        "a full symphony orchestra",
        "a solo pipe organ",
        "a choir singing",
        "an opera singer with orchestra",
        "a brass band",
    ],
    "mood": [
        "a sad and melancholic piece of music",
        "a joyful and cheerful piece of music",
        "a calm and peaceful piece of music",
        "a dramatic and tense piece of music",
        "a solemn and religious piece of music",
    ],
    "energy": [
        "a slow and quiet piece of music",
        "a fast and energetic piece of music",
    ],
}


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, dim=-1)


def embed_text(model: ClapModel, processor, labels: list[str], device: str) -> torch.Tensor:
    inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # transformers 5.x returns an output object; the joint-space vector
        # lives in pooler_output. Older code that used the return value
        # directly as a tensor breaks here.
        return normalize(model.get_text_features(**inputs).pooler_output)


def embed_audio(model: ClapModel, processor, path: Path, device: str) -> tuple[np.ndarray, float]:
    """Return (512-d track vector, noise_score).

    The track vector is the mean of its 10-second windows. Averaging the
    windows rather than truncating to the first 10 seconds matters here:
    a 78rpm side often opens with several seconds of lead-in noise.

    noise_score is spectral flatness, where 0 is tonal and 1 is white noise.
    Measured at 22 kHz on purpose: at the full 48 kHz the band above ~10 kHz
    is empty on shellac transfers, which drags every score toward zero and
    destroys the spread that makes the number useful.
    """
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    window = WINDOW_SECONDS * SAMPLE_RATE
    minimum = MIN_WINDOW_SECONDS * SAMPLE_RATE
    windows = [audio[i : i + window] for i in range(0, len(audio), window)]
    windows = [w for w in windows if len(w) >= minimum]
    if not windows:
        raise ValueError("audio too short")

    inputs = processor(audio=windows, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    with torch.no_grad():
        per_window = normalize(model.get_audio_features(**inputs).pooler_output)

    track = normalize(per_window.mean(dim=0, keepdim=True))[0]
    for_noise = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=22050)
    noise_score = float(librosa.feature.spectral_flatness(y=for_noise).mean())
    return track.cpu().numpy().astype(np.float32), noise_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"manifest holds {len(manifest)} tracks")

    # Resume: keep whatever was embedded on a previous run.
    vectors: dict[str, np.ndarray] = {}
    extras: dict[str, dict] = {}
    if EMBEDDINGS_PATH.exists():
        stored = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        identifiers = list(stored["identifiers"])
        for i, identifier in enumerate(identifiers):
            vectors[str(identifier)] = stored["vectors"][i]
        extras = json.loads(str(stored["extras"]))
        print(f"resuming with {len(vectors)} already embedded")

    print(f"loading {MODEL_NAME} ...")
    model = ClapModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    label_vectors = {
        group: embed_text(model, processor, labels, device)
        for group, labels in LABEL_GROUPS.items()
    }

    for number, (identifier, entry) in enumerate(manifest.items(), start=1):
        if identifier in vectors:
            continue

        path = PROJECT_ROOT / entry["audio_path"]
        if not path.exists():
            print(f"[{number}/{len(manifest)}] missing file, skip: {identifier[:50]}")
            continue

        try:
            vector, noise_score = embed_audio(model, processor, path, device)
        except Exception as error:
            print(f"[{number}/{len(manifest)}] failed ({error}): {identifier[:50]}")
            continue

        # Store the score against every label, not just the winner. Which label
        # wins turns out to be sensitive to how the prompt is phrased, so the
        # argmax is a weak signal on this corpus; the full vector at least lets
        # you revisit the decision later without re-embedding the audio.
        tensor = torch.from_numpy(vector).to(device).unsqueeze(0)
        tags = {}
        for group, matrix in label_vectors.items():
            scores = (tensor @ matrix.T)[0].tolist()
            tags[group] = {
                label: round(float(score), 4)
                for label, score in zip(LABEL_GROUPS[group], scores)
            }

        vectors[identifier] = vector
        extras[identifier] = {"noise_score": round(noise_score, 5), "tags": tags}
        top = max(tags["ensemble"], key=tags["ensemble"].get)
        print(
            f"[{number}/{len(manifest)}] {top[:28]:30}"
            f" noise={noise_score:.3f}  {identifier[:40]}"
        )

    identifiers = list(vectors.keys())
    matrix = np.stack([vectors[i] for i in identifiers])
    np.savez_compressed(
        EMBEDDINGS_PATH,
        identifiers=np.array(identifiers),
        vectors=matrix,
        extras=json.dumps(extras),
    )
    print(f"\nsaved {matrix.shape[0]} vectors of dim {matrix.shape[1]} to {EMBEDDINGS_PATH.name}")


if __name__ == "__main__":
    main()
