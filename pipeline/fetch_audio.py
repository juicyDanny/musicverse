"""Download public-domain classical recordings from the Internet Archive.

Targets the 78rpm collection: shellac-era recordings, mostly pre-1930,
which are in the public domain in the US. Each item is roughly one side
of a record, so tracks are short (3-5 minutes) and often self-contained.

Writes audio to data/raw/ and a manifest to data/manifest.json.
Safe to re-run: already downloaded items are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.parse import quote
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

# Archive.org asks that scripts identify themselves.
HEADERS = {"User-Agent": "musicverse/0.1 (github.com/juicyDanny/musicverse)"}

# Be polite: one request per second is the informal norm.
REQUEST_DELAY = 1.0


def make_session() -> requests.Session:
    """A session that retries on the failures archive.org actually produces:
    read timeouts, 429 rate limiting, and 5xx from overloaded nodes.
    Backoff grows 2s, 4s, 8s between attempts.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = make_session()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

COMPOSERS = [
    "Bach",
    "Mozart",
    "Beethoven",
    "Tchaikovsky",
    "Rachmaninoff",
    "Debussy",
    "Sibelius",
    "Elgar",
    "Chopin",
    "Brahms",
]

# Fields worth keeping. 'instrument' is unevenly populated but free when present.
SEARCH_FIELDS = [
    "identifier",
    "title",
    "creator",
    "year",
    "date",
    "instrument",
    "collection",
]


def search_composer(composer: str, rows: int) -> list[dict]:
    """Return search results for one composer, newest metadata first."""
    params = {
        "q": f'collection:(78rpm) AND creator:({composer})',
        "rows": rows,
        "page": 1,
        "output": "json",
    }
    # requests encodes repeated keys as fl[]=a&fl[]=b, which is what the API wants.
    params_list = (
        list(params.items())
        + [("fl[]", f) for f in SEARCH_FIELDS]
        # Default ordering is by relevance, which clumps the sides of one work
        # together. Random gives a spread across the composer's catalogue.
        + [("sort[]", "random asc")]
    )

    response = SESSION.get(SEARCH_URL, params=params_list, timeout=(10, 60))
    response.raise_for_status()
    return response.json()["response"]["docs"]


def pick_audio_file(identifier: str) -> tuple[str, int] | None:
    """Find the best audio file in an item. Returns (filename, size) or None.
    Returns None on network failure too: one bad item must never kill the run.

    Prefers MP3 over FLAC: roughly a third of the size, and every embedding
    model resamples to 16 or 48 kHz anyway, so the extra fidelity is discarded.
    """
    try:
        response = SESSION.get(
            METADATA_URL.format(identifier=identifier), timeout=(10, 60)
        )
        response.raise_for_status()
        files = response.json().get("files", [])
    except (requests.RequestException, ValueError) as error:
        print(f"    metadata failed: {error}")
        return None

    for wanted in ("VBR MP3", "MP3", "Flac"):
        for file in files:
            if file.get("format") == wanted:
                return file["name"], int(file.get("size") or 0)
    return None


def download(identifier: str, filename: str, destination: Path) -> bool:
    """Stream one file to disk. Returns False on any HTTP error."""
    # Filenames on archive.org contain spaces, commas and question marks.
    # An unescaped "?" turns the rest of the path into a query string and 404s.
    url = DOWNLOAD_URL.format(identifier=identifier, filename=quote(filename))
    try:
        with SESSION.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            # Write to a temp name so an interrupted run never leaves a
            # truncated file that looks complete on the next pass.
            temp_path = destination.with_suffix(destination.suffix + ".part")
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    handle.write(chunk)
            temp_path.rename(destination)
        return True
    except requests.RequestException as error:
        print(f"    download failed: {error}")
        return False


def load_manifest() -> dict[str, dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-composer",
        type=int,
        default=5,
        help="how many recordings to fetch per composer (default: 5)",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=25.0,
        help="skip files larger than this, in megabytes (default: 25)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    print(f"manifest currently holds {len(manifest)} tracks\n")

    for composer in COMPOSERS:
        print(f"{composer}")
        try:
            results = search_composer(composer, args.per_composer)
        except requests.RequestException as error:
            print(f"  search failed: {error}")
            continue
        time.sleep(REQUEST_DELAY)

        for doc in results:
            identifier = doc["identifier"]

            if identifier in manifest:
                print(f"  skip (already have) {identifier}")
                continue

            audio = pick_audio_file(identifier)
            time.sleep(REQUEST_DELAY)
            if audio is None:
                print(f"  skip (no audio) {identifier}")
                continue

            filename, size = audio
            if size > args.max_mb * 1024 * 1024:
                print(f"  skip (too big, {size / 1e6:.0f} MB) {identifier}")
                continue

            extension = Path(filename).suffix or ".mp3"
            destination = RAW_DIR / f"{identifier}{extension}"

            print(f"  get {identifier} ({size / 1e6:.1f} MB)")
            if not download(identifier, filename, destination):
                continue
            time.sleep(REQUEST_DELAY)

            creator = doc.get("creator")
            manifest[identifier] = {
                "identifier": identifier,
                "title": doc.get("title"),
                "creator": creator if isinstance(creator, list) else [creator],
                "composer_query": composer,
                "year": doc.get("year"),
                "date": doc.get("date"),
                "instrument": doc.get("instrument"),
                "audio_path": str(destination.relative_to(PROJECT_ROOT)),
                "source_url": f"https://archive.org/details/{identifier}",
            }
            save_manifest(manifest)

    print(f"\ndone. manifest holds {len(manifest)} tracks")


if __name__ == "__main__":
    main()
