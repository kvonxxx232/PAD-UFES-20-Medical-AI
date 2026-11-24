"""Label-blinded, native-resolution PNG duplicate audit for protocol V1.1."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import os
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image
from scipy.fftpack import dct


@dataclass
class ImageAudit:
    confirmed: dict[tuple[str, str], str]
    candidates: dict[tuple[str, str], int]
    file_hashes: dict[str, str]


@dataclass(frozen=True)
class _ImageHashes:
    file_sha256: str
    rgb_key: tuple[tuple[int, int], bytes]
    phash: int


def phash64(image: Image.Image) -> int:
    """ImageHash phash: grayscale, Lanczos, separable unnormalised DCT-II."""
    small = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    coefficients = dct(dct(pixels, type=2, axis=0), type=2, axis=1)[:8, :8]
    bits = np.packbits(coefficients > np.median(coefficients), bitorder="big")
    return int.from_bytes(bits.tobytes(), "big")


def _inventory(image_keys, image_dirs):
    def walk_error(error):
        raise ValueError(f"Could not inventory image directory: {error}") from error

    paths = {}
    for directory in image_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Missing image directory: {directory}")
        for root, _, filenames in os.walk(directory, onerror=walk_error):
            for name in filenames:
                if not name.lower().endswith(".png"):
                    continue
                path = Path(root, name)
                if name in paths:
                    raise ValueError(f"Ambiguous PNG filename: {paths[name]} and {path}")
                paths[name] = path
    missing = sorted(image_keys.keys() - paths.keys())
    extra = sorted(paths.keys() - image_keys.keys())
    if missing or extra:
        raise ValueError(f"PNG inventory mismatch. Missing: {missing!r}. Extra: {extra!r}")
    return paths


def _read(path):
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read PNG {path}: {error}") from error


def _hash_image(path):
    data = _read(path)
    try:
        with Image.open(BytesIO(data), formats=("PNG",)) as image:
            image.verify()
        # verify() closes the PNG decoder. Decode and hash the same file bytes.
        with Image.open(BytesIO(data), formats=("PNG",)) as image:
            if image.n_frames != 1:
                raise ValueError("Multiple frames in a single metadata image")
            image.load()
            rgb = image.convert("RGB")
            rgb_key = (rgb.size, sha256(rgb.tobytes()).digest())
            return _ImageHashes(sha256(data).hexdigest(), rgb_key, phash64(image))
    except (OSError, ValueError, SyntaxError, EOFError, struct.error, zlib.error,
            Image.DecompressionBombError) as error:
        raise ValueError(f"Unreadable or unsupported PNG {path}: {error}") from error


def audit_images(
    image_keys: dict[str, tuple[str, str]], image_dirs: list[Path]
) -> ImageAudit:
    """Use only filenames, key membership and image content; retain no pixel arrays."""
    paths = _inventory(image_keys, image_dirs)
    names = sorted(paths)
    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes = dict(zip(names, pool.map(_hash_image, (paths[name] for name in names))))

    by_file = defaultdict(list)
    by_rgb = defaultdict(list)
    for name in names:
        by_file[hashes[name].file_sha256].append(name)
        by_rgb[hashes[name].rgb_key].append(name)

    def checked_bytes(name):
        data = _read(paths[name])
        if sha256(data).hexdigest() != hashes[name].file_sha256:
            raise ValueError(f"PNG changed during audit: {name}")
        return data

    def checked_pixels(name):
        with Image.open(BytesIO(checked_bytes(name)), formats=("PNG",)) as image:
            return image.size, image.convert("RGB").tobytes()

    confirmed = {}
    for groups, read_content, rule in (
        (by_file, checked_bytes, "sha256"),
        (by_rgb, checked_pixels, "rgb_equal"),
    ):
        for group in groups.values():
            for a, b in combinations(group, 2):
                if image_keys[a] == image_keys[b]:
                    continue
                if read_content(a) != read_content(b):
                    raise ValueError(f"{rule} digest match failed content comparison: {a}, {b}")
                # Each file-byte match is also an RGB match. Count it once.
                confirmed.setdefault((a, b), rule)

    candidates = {}
    for a, b in combinations(names, 2):
        if image_keys[a] == image_keys[b] or hashes[a].file_sha256 == hashes[b].file_sha256:
            continue
        distance = (hashes[a].phash ^ hashes[b].phash).bit_count()
        if distance <= 8:
            candidates[a, b] = distance

    return ImageAudit(
        confirmed=dict(sorted(confirmed.items())),
        candidates=candidates,
        file_hashes={name: hashes[name].file_sha256 for name in names},
    )
