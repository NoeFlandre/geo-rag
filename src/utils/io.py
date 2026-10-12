"""Compatibility facade for the I/O helpers.

The implementations now live in focused modules: table_io, clusters,
embeddings, offline_images, wildlife_insights and image_download.
Existing imports from this module keep working.
"""

from src.utils.clusters import load_dataset_with_clusters
from src.utils.companion_paths import get_core_base_name
from src.utils.embeddings import load_embeddings
from src.utils.image_download import _get_http_session, download_image
from src.utils.offline_images import (
    KNOWN_PLACEHOLDER_MD5_HASHES,
    build_offline_image_index,
    is_valid_image_file,
    resolve_offline_image_path,
)
from src.utils.table_io import get_parquet_writer, load_dataframe, save_dataframe
from src.utils.wildlife_insights import (
    get_wildlife_insights_token,
    resolve_wildlife_insights_url,
)

__all__ = [
    "KNOWN_PLACEHOLDER_MD5_HASHES",
    "_get_http_session",
    "build_offline_image_index",
    "download_image",
    "get_core_base_name",
    "get_parquet_writer",
    "get_wildlife_insights_token",
    "is_valid_image_file",
    "load_dataframe",
    "load_dataset_with_clusters",
    "load_embeddings",
    "resolve_offline_image_path",
    "resolve_wildlife_insights_url",
    "save_dataframe",
]
