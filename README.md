# musicverse

An explorable 3D map of classical recordings, positioned by how they actually sound.

Every recording is turned into a 512-dimensional vector by an audio embedding model,
then projected down to three dimensions. Recordings that sound alike end up near each
other. The result is a space you can fly through, where distance means something.

**Status: early.** The data pipeline works end to end on a small corpus. The 3D
frontend is not built yet. What exists today is the machinery that produces the map,
plus a set of measurements about whether the map is meaningful — which turned out to
be the more interesting question.

---

## What the data says so far

These are results from an actual run, not aspirations. The current corpus is 46
recordings drawn from the Internet Archive's 78rpm collection.

### The map organises by timbre, not by composition

The corpus happened to contain two recordings of Debussy's *Clair de Lune*: one by
Leopold Stokowski with the Philadelphia Orchestra, one by Moura Lympany on solo piano.

They did not find each other. The Stokowski recording's closest match was Elgar's
*Enigma Variations*. The Lympany recording's closest match was a Chopin étude played
by Alfred Cortot. Both pairs scored 0.863.

The same composition landed in opposite regions of the space depending on what
instrument played it. The pattern repeats across the corpus — a Mozart aria paired
with a Rachmaninoff choral piece, two works with nothing in common except a solo
voice over accompaniment.

So this map has continents of piano, of orchestra, of violin, of voice. It does not
have a continent of Debussy. That is a real property of the space, worth knowing
before reading anything into a cluster.

### Recording noise shapes the space, asymmetrically

Each track carries a `noise_score` (spectral flatness: 0 is tonal, 1 is white noise).
Splitting the corpus at the median and comparing similarities:

| | mean cosine similarity |
|---|---|
| within the noisy half | 0.836 |
| within the clean half | 0.773 |
| across the two halves | 0.752 |

The halves are not two symmetric clusters. The clean half sits barely above the
cross-half baseline; the noisy half sits well above it. Noisy transfers collapse onto
each other. When broadband hiss dominates the spectrum there is less musical signal
left to tell recordings apart.

The seven acoustic-era Edison discs in the corpus all scored in the upper half.

### Zero-shot text labels are not usable here

CLAP shares a space between audio and text, so in principle you can label a recording
by comparing it against prompts. In practice the winning label flips depending on how
the prompt is phrased. Four phrasings of the same instrument list were tried; with one
set, 30 of 46 tracks were labelled "a full symphony orchestra", including a solo piano
étude. A Bach organ prelude was called piano, orchestra, or organ depending on wording.

The audio vectors are sound. The alignment with text is not, on hundred-year-old
transfers. Labels are stored as full score vectors rather than a winning label, so the
decision can be revisited without re-processing audio.

### The projection is faithful enough to build on

Trustworthiness measures how much of the 512-dimensional neighbourhood survives the
trip to 3D. Uncentred, it scores 0.885. Mean-centring the vectors first raises it to
**0.955** — the same structure, spread out over a wider range of distances, which is
easier to preserve in three dimensions. The ranked list of most-similar pairs is
nearly identical either way.

---

## How it works

```
Internet Archive 78rpm  ->  fetch_audio.py  ->  data/raw/*.mp3
                                                data/manifest.json
                                     |
                                     v
         CLAP embeddings   ->  embed.py     ->  data/embeddings.npz
      (512-d, 10s windows,                      512-d vector per track
       mean-pooled)                             + noise score + label scores
                                     |
                                     v
         UMAP to 3D        ->  project.py   ->  docs/data/universe.json
                                                coordinates, metadata,
                                                precomputed neighbours
                                     |
                                     v
                              (frontend, not built yet)
```

All the heavy work happens offline. The published site is static: it loads one JSON
file and renders it. No server, no database.

**No vector database.** At this scale it would be infrastructure without a purpose.
Nearest neighbours are computed offline and shipped inside the JSON. That changes if
the corpus reaches six figures, or if live audio queries are added.

**Neighbours come from the 512-d space, not from the 3D picture.** Projections
distort distance beyond the immediate neighbourhood, so "similar recordings" in the
interface must never be read off the map.

---

## Layout

```
pipeline/          Python, runs offline
  fetch_audio.py     download from the Internet Archive
  embed.py           audio -> CLAP vectors
  project.py         vectors -> 3D coordinates, plus diagnostics
docs/              the published static site (GitHub Pages serves this folder)
  data/
    universe.json    the map: coordinates, metadata, neighbours
data/              local working files, not in the repo
  raw/               downloaded audio
  manifest.json      what was downloaded
  embeddings.npz     the vectors
```

Audio never enters the repository. Git handles large binaries badly and keeps them in
history forever.

---

## Running it

Requires Python 3.10+, ffmpeg, and a CUDA GPU for the embedding step (CPU works, slowly).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pipeline/requirements.txt

python pipeline/fetch_audio.py --per-composer 5   # download
python pipeline/embed.py                          # embed
python pipeline/project.py --center --plot        # project + diagnostics
```

Every stage is resumable and skips work already done, so re-running is cheap. Because
search results are randomised, running the fetcher repeatedly accumulates variety
rather than the same recordings.

`project.py` prints diagnostics rather than only producing a picture. UMAP will always
draw something; the numbers are what tell you whether it means anything.

---

## Data and licensing

Audio comes from the Internet Archive's 78rpm collection. Metadata there is uneven in
ways that matter:

- The `creator` field mixes composers with performers and does not distinguish them.
  One disc is credited to both Rachmaninoff and Chopin — it is Rachmaninoff *playing*
  Chopin. Composer attribution in this project is therefore a search artefact, not a
  fact, and is not used as a label.
- One item is one side of a record, roughly four minutes. Longer works are split
  across several items. A track here is not a work.

On copyright: under the Music Modernization Act, US sound recordings published through
1925 are in the public domain, with the line advancing one year each January. Parts of
the 78rpm collection are more recent than that.

**This repository redistributes no audio.** It contains derived coordinates, metadata,
and links back to the original Internet Archive items. Audio downloaded locally stays
local.

The code is MIT licensed. That covers the code only.

---

## Roadmap

- [x] Fetch public-domain recordings with resumable downloads
- [x] CLAP embeddings with a per-track recording-noise score
- [x] UMAP projection with quality diagnostics
- [ ] Three.js frontend: fly through the map, click a point, see its neighbours
- [ ] Grow the corpus past 150 recordings so UMAP has more to work with
- [ ] Cross-reference MusicBrainz for real composer and period metadata
- [ ] Colour and filter by recording noise, to make the artefact visible instead of hidden
- [ ] Audio preview on click, restricted to recordings clearly in the public domain
- [ ] Melody and harmony features from MIDI or automatic transcription — CLAP does not
      represent pitch explicitly, and more data will not fix that

---

## Why the awkward folder names

`docs/` holds the frontend rather than documentation because GitHub Pages publishes
from either the repository root or a folder with that exact name. It is a small price
for a zero-configuration deploy.
