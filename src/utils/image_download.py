"""Image download over HTTP with a shared session, offline fallbacks and retries."""

import os
import time
from io import BytesIO

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.utils.offline_images import is_valid_image_file, resolve_offline_image_path
from src.utils.wildlife_insights import resolve_wildlife_insights_url

# Thread-safe global session for download adapters
_http_session = None


def _get_http_session():
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=Retry(
                total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
            ),
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


def download_image(
    url,
    mapillary_token,
    photo_id=None,
    platform=None,
    offline_dirs=None,
    image_size=448,
    max_retries=3,
):
    """Loads an image locally if it's part of an offline dataset, or downloads it via connection pool with retries and fallbacks."""
    for attempt in range(max_retries):
        try:
            # 1. Try local direct path
            if url and os.path.exists(url):
                img = Image.open(url).convert("RGB")
                img_resized = img.resize((image_size, image_size))
                return img_resized

            # 2. Try resolving via offline directories
            dirs_to_use = offline_dirs if offline_dirs is not None else []
            if dirs_to_use and url:
                resolved = resolve_offline_image_path(
                    url, dirs_to_use, photo_id, platform
                )
                if resolved and os.path.exists(resolved):
                    img = Image.open(resolved).convert("RGB")
                    img_resized = img.resize((image_size, image_size))
                    return img_resized

            # 3. Fallback to download over HTTP
            if url:
                session = _get_http_session()
                # Mapillary schema resolution
                if url.startswith("mapillary://") or (photo_id and "fbcdn.net" in url):
                    orig_id = str(photo_id) if photo_id else url.split("://")[1]
                    api_url = (
                        f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
                    )
                    headers = {"Authorization": f"OAuth {mapillary_token}"}
                    res = session.get(api_url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        url = res.json().get("thumb_1024_url")
                # KartaView schema resolution
                elif url.startswith("kartaview://"):
                    orig_id = url.split("://")[1]
                    api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
                    res = session.get(api_url, timeout=10)
                    if res.status_code == 200:
                        data = res.json().get("result", {}).get("data", {})
                        url = (
                            data.get("fileurlLTh")
                            or data.get("fileurlTh")
                            or data.get("fileurl")
                        )
                # Wildlife Insights schema resolution
                elif "wildlifeinsights.org" in url or (
                    platform
                    and str(platform).lower()
                    in ["snapshotusa", "snapshot_usa", "wildlife_insights"]
                ):
                    url = resolve_wildlife_insights_url(url, photo_id=photo_id)

                if url:
                    res = session.get(url, timeout=10)
                    if res.status_code == 200:
                        c_type = res.headers.get("content-type", "").lower()
                        if "text/html" not in c_type and is_valid_image_file(
                            res.content
                        ):
                            img = Image.open(BytesIO(res.content)).convert("RGB")
                            img_resized = img.resize((image_size, image_size))
                            return img_resized
        except Exception:
            pass

        # If it fails, wait and try schema fallback if applicable
        if attempt < max_retries - 1:
            time.sleep(1.0 * (attempt + 1))
            if platform:
                plat_lower = platform.lower()
                if (
                    plat_lower == "mapillary"
                    and url
                    and not url.startswith("mapillary://")
                ):
                    url = f"mapillary://{photo_id}"
                elif (
                    plat_lower == "kartaview"
                    and url
                    and not url.startswith("kartaview://")
                ):
                    url = f"kartaview://{photo_id}"

    return None
