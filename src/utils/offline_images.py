"""Local image checks and lookup for offline image datasets."""

import hashlib
import os

# Known censor/placeholder image hashes (e.g. Wildlife Insights "Human in frame" placeholder)
KNOWN_PLACEHOLDER_MD5_HASHES = {
    "6da31bd91c242aa98de0ddae4ad89e3c",  # Wildlife Insights 1600x1200 "Human in frame" PNG placeholder (8,535 bytes)
}


def is_valid_image_file(filepath_or_bytes, reject_placeholders=True):
    """Checks if a file path or raw bytes contains valid image data (JPEG, PNG, GIF, WebP, TIFF)."""
    full_bytes = None
    file_path = None
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        data = filepath_or_bytes[:32]
        full_bytes = filepath_or_bytes
    elif isinstance(filepath_or_bytes, (str, os.PathLike)):
        file_path = filepath_or_bytes
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 100:
            return False
        try:
            with open(file_path, "rb") as f:
                data = f.read(32)
        except Exception:
            return False
    else:
        return False

    if not data or len(data) < 4:
        return False

    # Check for known censor/placeholder images
    if reject_placeholders:
        try:
            h = None
            if full_bytes is not None:
                h = hashlib.md5(full_bytes).hexdigest()
            elif file_path is not None:
                if os.path.getsize(file_path) == 8535:
                    with open(file_path, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
            if h and h in KNOWN_PLACEHOLDER_MD5_HASHES:
                return False
        except Exception:
            pass

    # Check magic bytes for standard image formats
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):  # GIF
        return True
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":  # WebP
        return True
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):  # TIFF
        return True

    return False


def build_offline_image_index(image_root_dirs, extensions=None):
    """
    Recursively scans image_root_dirs to build a fast in-memory lookup index
    supporting arbitrary nested directory structures.

    Returns a dict mapping keys (exact filenames, stems, relative subpaths,
    and platform-qualified stems) to absolute file paths on disk.
    """
    if not image_root_dirs:
        return {}

    dirs = [image_root_dirs] if isinstance(image_root_dirs, str) else image_root_dirs
    if extensions is None:
        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tif",
            ".tiff",
            ".gif",
            ".bmp",
        }

    index = {}

    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        abs_d = os.path.abspath(d)
        for root, _, files in os.walk(abs_d):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in extensions:
                    continue
                full_path = os.path.join(root, fn)
                stem = os.path.splitext(fn)[0]
                rel_path = os.path.relpath(full_path, abs_d)

                # Store direct filename and stem
                for k in [fn, fn.lower(), stem, stem.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Store relative subpath
                rel_clean = rel_path.replace("\\", "/")
                for k in [rel_clean, rel_clean.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Subpath components (e.g. "site_A/deployment_1/photo_999.JPG")
                parts = rel_clean.split("/")
                if len(parts) > 1:
                    sub_key = "/".join(parts[-2:])
                    if sub_key not in index:
                        index[sub_key] = full_path
                    if sub_key.lower() not in index:
                        index[sub_key.lower()] = full_path

                    stem_sub_key = f"{parts[-2]}/{stem}"
                    if stem_sub_key not in index:
                        index[stem_sub_key] = full_path
                    if stem_sub_key.lower() not in index:
                        index[stem_sub_key.lower()] = full_path

    return index


def resolve_offline_image_path(
    url, image_root_dirs, photo_id=None, platform=None, image_index=None
):
    """
    Resolves an image URL/ID to a local path on disk by checking an optional
    precomputed recursive image_index, flat files, train/ folders, and nested
    directory structures.
    Returns the absolute path if found, otherwise None.
    """
    # 0. Check precomputed recursive image_index first if provided
    if image_index is not None:
        # A. Try platform + photo_id
        if photo_id:
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]

            # Fast direct ID check in index
            if photo_str in image_index:
                return image_index[photo_str]
            photo_str_lower = photo_str.lower()
            if photo_str_lower in image_index:
                return image_index[photo_str_lower]

            if platform:
                plat_str = str(platform).strip().lower()
                plat_key = f"{plat_str}/{photo_str_lower}"
                if plat_key in image_index:
                    return image_index[plat_key]
                for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                    candidate = f"{plat_key}{ext.lower()}"
                    if candidate in image_index:
                        return image_index[candidate]

            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                candidate = f"{photo_str}{ext}"
                if candidate in image_index:
                    return image_index[candidate]
                if candidate.lower() in image_index:
                    return image_index[candidate.lower()]

        # B. Try url
        if url and isinstance(url, str):
            clean_url = url.split("://")[1] if "://" in url else url
            clean_url = clean_url.strip()
            if clean_url in image_index:
                return image_index[clean_url]
            clean_url_lower = clean_url.lower()
            if clean_url_lower in image_index:
                return image_index[clean_url_lower]

            url_no_params = clean_url.split("?")[0].split("#")[0]
            if url_no_params != clean_url:
                if url_no_params in image_index:
                    return image_index[url_no_params]
                if url_no_params.lower() in image_index:
                    return image_index[url_no_params.lower()]

            basename = os.path.basename(url_no_params)
            if basename in image_index:
                return image_index[basename]
            basename_lower = basename.lower()
            if basename_lower in image_index:
                return image_index[basename_lower]

            stem = os.path.splitext(basename)[0]
            if stem in image_index:
                return image_index[stem]
            stem_lower = stem.lower()
            if stem_lower in image_index:
                return image_index[stem_lower]

            parts = url_no_params.replace("\\", "/").split("/")
            if len(parts) > 1:
                sub_key = "/".join(parts[-2:])
                if sub_key in image_index:
                    return image_index[sub_key]
                if sub_key.lower() in image_index:
                    return image_index[sub_key.lower()]

        # When precomputed image_index is provided, image_root_dirs have been fully indexed.
        # Avoid expensive fall-through disk checks on non-matching records.
        return None

    if not image_root_dirs:
        return None

    dirs = [image_root_dirs] if isinstance(image_root_dirs, str) else image_root_dirs
    for d in dirs:
        if not d:
            continue

        # 0. Try direct relative path join first
        p_direct = os.path.join(d, url) if url else None
        if p_direct and os.path.exists(p_direct):
            return os.path.abspath(p_direct)

        # A. Try direct lookup using Photo_ID (flat file, train/ folder, or nested)
        if photo_id:
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                # 1. Platform-specific subfolder lookup (e.g. d/platform/photo_id.jpg)
                if platform:
                    plat_str = str(platform).strip().lower()
                    p_plat = os.path.join(d, plat_str, f"{photo_str}{ext}")
                    if os.path.exists(p_plat):
                        return os.path.abspath(p_plat)

                # 2. Flat lookup
                p_flat = os.path.join(d, f"{photo_str}{ext}")
                if os.path.exists(p_flat):
                    return os.path.abspath(p_flat)

                # 2. train/ flat lookup
                p_train = os.path.join(d, "train", f"{photo_str}{ext}")
                if os.path.exists(p_train):
                    return os.path.abspath(p_train)

                # 3. Nested GLDv2 lookup (e.g. d/a/b/c/id.jpg)
                if len(photo_str) == 16:
                    p_nested = os.path.join(
                        d, photo_str[0], photo_str[1], photo_str[2], f"{photo_str}{ext}"
                    )
                    if os.path.exists(p_nested):
                        return os.path.abspath(p_nested)
                    p_nested_train = os.path.join(
                        d,
                        "train",
                        photo_str[0],
                        photo_str[1],
                        photo_str[2],
                        f"{photo_str}{ext}",
                    )
                    if os.path.exists(p_nested_train):
                        return os.path.abspath(p_nested_train)

        # B. Fallback to URL basename lookup
        if url and isinstance(url, str):
            clean_url = url
            if "://" in url:
                clean_url = url.split("://")[1]
            basename = os.path.basename(clean_url)

            basenames = [basename]
            if "." not in basename:
                basenames.extend(
                    [
                        f"{basename}{ext}"
                        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]
                    ]
                )

            for b in basenames:
                p_base = os.path.join(d, b)
                if os.path.exists(p_base):
                    return os.path.abspath(p_base)
                p_base_train = os.path.join(d, "train", b)
                if os.path.exists(p_base_train):
                    return os.path.abspath(p_base_train)

    return None
