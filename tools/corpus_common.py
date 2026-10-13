"""Small offline corpus contracts; no overwrite and no implicit network access."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

SEP = "\n<|doc|>\n"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_texts(texts, digest=None):
    """Digest buckets are an index; exact text equality decides membership."""
    digest = digest or (lambda text: hashlib.sha256(text.encode("utf-8")).digest())
    seen = {}
    result = []
    for text in texts:
        bucket = seen.setdefault(digest(text), [])
        if text not in bucket:
            bucket.append(text)
            result.append(text)
    return result


def split_documents(documents, seed=42, validation_percent=1):
    if not 0 < validation_percent < 100:
        raise ValueError("validation_percent must be between 0 and 100")
    if len(documents) < 2:
        raise ValueError("at least two documents are required for nonempty splits")
    documents = list(documents)
    random.Random(seed).shuffle(documents)
    cut = min(len(documents) - 1, max(1, len(documents) * validation_percent // 100))
    return documents[cut:], documents[:cut]


def publish_files(outputs):
    """Stage complete bytes, fsync, then hard-link each destination exclusively.

    Requires hard-link support on the output filesystem; never falls back to
    overwrite. This is per-file publication, NOT a multi-file transaction or a
    power-loss durability guarantee. On a later failure, already published
    complete files remain; only uniquely created temporary paths are cleaned.
    """
    staged = []
    try:
        for destination, content in outputs:
            destination = Path(destination)
            fd, temporary = tempfile.mkstemp(prefix=".niyah-corpus-", dir=destination.parent)
            staged.append((Path(temporary), destination))
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.link(temporary, destination)
    finally:
        failures = []
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except OSError as exc:
                failures.append(f"{temporary}: {exc}")
        if failures:
            raise OSError("temporary cleanup failed: " + "; ".join(failures))


def publish_corpus(outputs, provenance):
    """Publish data followed by a hash-bound provenance record per output."""
    publications = []
    for path, documents in outputs:
        if not documents or any(not isinstance(t, str) or not t.strip() for t in documents):
            raise ValueError("refusing empty or invalid corpus")
        payload = "".join(t + SEP for t in documents).encode("utf-8")
        metadata = dict(provenance, schema_version=1, documents=len(documents),
                        characters=sum(map(len, documents)), bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        budget_units="characters, unless explicitly specified otherwise",
                        split_units="documents; not tokens", realized_token_mixture="not measured")
        publications.append((path, payload))
        publications.append((str(path) + ".provenance.json",
                             (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")))
    for path, _ in publications:
        if os.path.lexists(path):
            raise FileExistsError(path)
    publish_files(publications)
