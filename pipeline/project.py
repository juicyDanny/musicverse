"""Project CLAP embeddings down to 3D and export the map data.

Reads data/embeddings.npz and data/manifest.json.
Writes docs/data/universe.json, which is what the frontend will load.

Also prints diagnostics, because the interesting question is not whether
UMAP produces a picture (it always does) but whether the picture is about
music or about recording noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import trustworthiness

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "universe.json"

NEIGHBOURS_TO_STORE = 10


def load() -> tuple[list[str], np.ndarray, dict, dict]:
    stored = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    identifiers = [str(x) for x in stored["identifiers"]]
    vectors = stored["vectors"].astype(np.float64)
    extras = json.loads(str(stored["extras"]))
    manifest = json.loads(MANIFEST_PATH.read_text())
    return identifiers, vectors, extras, manifest


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def noise_diagnostic(similarity: np.ndarray, noise: np.ndarray) -> None:
    """Split the corpus at the median noise score and compare similarities.

    If tracks resemble each other mainly because they were transferred from
    the same kind of disc, the two halves will each be internally tight and
    mutually distant. If the space is about music, the split should barely
    matter.
    """
    median = float(np.median(noise))
    loud = noise > median
    quiet = ~loud

    def mean_block(rows: np.ndarray, columns: np.ndarray) -> float:
        block = similarity[np.ix_(rows, columns)]
        if rows is columns or np.array_equal(rows, columns):
            block = block[~np.eye(block.shape[0], dtype=bool)]
        return float(block.mean())

    loud_idx = np.where(loud)[0]
    quiet_idx = np.where(quiet)[0]
    within_loud = mean_block(loud_idx, loud_idx)
    within_quiet = mean_block(quiet_idx, quiet_idx)
    across = float(similarity[np.ix_(loud_idx, quiet_idx)].mean())
    gap = (within_loud + within_quiet) / 2 - across

    print(f"\n  median noise score: {median:.3f}")
    print(f"  similarity within the noisy half : {within_loud:.3f}")
    print(f"  similarity within the clean half : {within_quiet:.3f}")
    print(f"  similarity across the two halves : {across:.3f}")
    print(f"  gap: {gap:+.3f}")
    if gap > 0.05:
        print("  -> recording condition is shaping the space. Expect the map")
        print("     to separate transfers as much as it separates music.")
    elif gap > 0.02:
        print("  -> mild effect. Visible if you colour by noise, not dominant.")
    else:
        print("  -> negligible. The space is not organised by recording noise.")


def top_pairs(identifiers: list[str], similarity: np.ndarray, count: int = 8) -> None:
    """The most similar pairs in the corpus. Read these by eye: sides of the
    same work landing together is the strongest evidence the space is sane.
    """
    working = similarity.copy()
    np.fill_diagonal(working, -1)
    flat = np.triu(working, k=1)
    order = np.dstack(np.unravel_index(np.argsort(flat.ravel())[::-1], flat.shape))[0]
    print()
    for i, j in order[:count]:
        print(f"  {flat[i][j]:.3f}  {identifiers[i][:44]:46} <-> {identifiers[j][:44]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--center",
        action="store_true",
        help="subtract the corpus mean before projecting; spreads out a "
        "corpus whose vectors are all crowded together",
    )
    parser.add_argument("--neighbors", type=int, default=None, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.15, help="UMAP min_dist")
    parser.add_argument("--plot", action="store_true", help="also save a PNG preview")
    args = parser.parse_args()

    identifiers, vectors, extras, manifest = load()
    count = len(identifiers)
    noise = np.array([extras[i]["noise_score"] for i in identifiers])
    print(f"{count} tracks, {vectors.shape[1]} dimensions")
    print(f"noise score: min {noise.min():.3f}  max {noise.max():.3f}  mean {noise.mean():.3f}")

    working = vectors - vectors.mean(axis=0) if args.center else vectors
    working = unit(working)
    similarity = working @ working.T

    print("\n=== most similar pairs (512-d space) ===")
    top_pairs(identifiers, similarity)

    print("\n=== is the space organised by recording noise? ===")
    noise_diagnostic(similarity, noise)

    # n_neighbors must stay below the corpus size, and small corpora need a
    # small value or UMAP smears everything into one blob.
    n_neighbors = args.neighbors or max(2, min(15, count - 1))

    import umap  # imported late: it is slow to load and pulls in numba

    print(f"\n=== UMAP -> 3D (n_neighbors={n_neighbors}, min_dist={args.min_dist}) ===")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        # Fixed seed so the map does not rearrange itself on every run.
        random_state=42,
    )
    coordinates = reducer.fit_transform(working)

    # sklearn requires fewer neighbours than half the corpus.
    k = max(2, min(10, (count - 1) // 2))
    score = trustworthiness(working, coordinates, n_neighbors=k)
    print(f"  trustworthiness (k={k}): {score:.3f}")
    print("  (1.0 means every close pair in 512-d stayed close in 3-d;")
    print("   below about 0.85 the picture is misleading at the local level)")

    # Centre on the origin and scale to a fixed radius so the frontend camera
    # does not need to know anything about the data.
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = coordinates / np.abs(coordinates).max() * 50

    neighbour_order = np.argsort(-similarity, axis=1)

    tracks = []
    for index, identifier in enumerate(identifiers):
        entry = manifest.get(identifier, {})
        neighbours = [
            {"id": identifiers[j], "score": round(float(similarity[index][j]), 4)}
            for j in neighbour_order[index][1 : NEIGHBOURS_TO_STORE + 1]
        ]
        tracks.append(
            {
                "id": identifier,
                "title": entry.get("title"),
                "creator": entry.get("creator"),
                "year": entry.get("year"),
                "x": round(float(coordinates[index][0]), 3),
                "y": round(float(coordinates[index][1]), 3),
                "z": round(float(coordinates[index][2]), 3),
                "noise": extras[identifier]["noise_score"],
                "source_url": entry.get("source_url"),
                # Neighbours come from the 512-d space, not from the 3-d
                # picture. Distances in the projection are not trustworthy
                # beyond the immediate neighbourhood, so "similar tracks" in
                # the UI must not be read off the map.
                "neighbours": neighbours,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"tracks": tracks}, indent=1, ensure_ascii=False)
    )
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nwrote {len(tracks)} tracks to {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(9, 8))
        axes = figure.add_subplot(111, projection="3d")
        dots = axes.scatter(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            c=noise, cmap="viridis", s=60, depthshade=False,
        )
        figure.colorbar(dots, label="noise score")
        axes.set_title("Coloured by recording noise: look for a colour-sorted map")
        path = PROJECT_ROOT / "data" / "preview.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        print(f"saved preview to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
