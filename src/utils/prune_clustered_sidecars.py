#!/usr/bin/env python3
"""
Prunes redundant base dataset metadata columns from clustered Parquet sidecar files.

In Geo-RAG's decoupled architecture, clustered files are intended to be lightweight
sidecars (~150 MB) containing only primary keys (photo_key, Platform, Photo_ID),
cluster assignments (cluster_id, parent_cluster_id), and VLM text descriptions/labels.

If a script mistakenly re-serializes the base metadata (Latitude, Longitude, Image_URL, etc.)
into the sidecar, this utility strips the redundant columns using memory-efficient PyArrow
streaming, shrinking files by ~95% and restoring single-source-of-truth metadata integrity.
"""

import argparse
import glob
import os
import stat
import sys
import time
from typing import Dict, List, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import get_parquet_writer

CANONICAL_SIDECAR_COLS: Set[str] = {
    "photo_key",
    "Platform",
    "Photo_ID",
    "embedding_idx",
    "cluster_id",
    "parent_cluster_id",
    "cluster_label",
    "cluster_description",
    "parent_cluster_label",
    "parent_cluster_description",
    "visual_description",
    "parent_visual_description",
    "model_name",
    "representation_type",
    "precision",
}

REDUNDANT_BASE_COLS: Set[str] = {
    "Latitude",
    "Longitude",
    "Image_URL",
    "image_url",
    "url",
    "source",
    "Captured_At",
    "Date_Taken",
    "date_taken",
    "H3_Cell",
    "h3_cell",
    "h3_index",
    "Season",
    "Time_Of_Day",
    "Koppen_Code",
    "Koppen_Desc",
    "continent",
    "country",
    "License",
    "License_Code",
    "License_Name",
    "User_Tags",
    "Title",
    "Description",
    "embedding",
    "patch_embedding",
}


def is_sidecar_column(col_name: str) -> bool:
    """Returns True if the column belongs to a clustered sidecar."""
    if col_name in CANONICAL_SIDECAR_COLS:
        return True
    if col_name in REDUNDANT_BASE_COLS:
        return False
    lower = col_name.lower()
    return "cluster" in lower or "medoid" in lower or lower.endswith("_description")


def find_clustered_parquet_files(
    targets: List[str], recursive: bool = False, match_all_parquets: bool = False
) -> List[str]:
    """Discovers all clustered parquet files from input paths/directories."""
    discovered: Set[str] = set()

    for target in targets:
        if not os.path.exists(target):
            print(f"[WARNING] Path does not exist: {target}", file=sys.stderr)
            continue

        if os.path.isfile(target):
            if (
                target.endswith(".parquet")
                and not target.endswith(".tmp")
                and not target.endswith(".keys.parquet")
            ):
                discovered.add(os.path.abspath(target))
            continue

        # Target is a directory
        pattern = "**/*.parquet" if recursive else "*.parquet"
        for p in glob.glob(os.path.join(target, pattern), recursive=recursive):
            if p.endswith(".keys.parquet") or ".tmp" in p or "_h3_semantic_index" in p:
                continue

            fname = os.path.basename(p)
            if match_all_parquets or "_clustered_" in fname or "_clustered_k_" in fname:
                discovered.add(os.path.abspath(p))

    return sorted(list(discovered))


def inspect_sidecar_schema(file_path: str) -> Tuple[List[str], List[str], bool]:
    """
    Inspects schema of a parquet file.
    Returns: (cols_to_keep, cols_to_drop, is_valid_sidecar)
    """
    pf = pq.ParquetFile(file_path)
    all_cols = pf.schema_arrow.names

    # Check if this file has cluster assignments
    has_cluster = any("cluster_id" in c for c in all_cols)
    has_key = any(c in all_cols for c in ["photo_key", "Photo_ID", "Platform"])

    if not has_cluster or not has_key:
        return all_cols, [], False

    cols_to_keep = [c for c in all_cols if is_sidecar_column(c)]
    cols_to_drop = [c for c in all_cols if c not in cols_to_keep]

    return cols_to_keep, cols_to_drop, True


def prune_single_sidecar(
    file_path: str,
    dry_run: bool = False,
    backup: bool = False,
) -> Dict[str, object]:
    """
    Prunes redundant columns from a single clustered parquet file using Arrow streaming.
    """
    start_time = time.time()
    orig_size_bytes = os.path.getsize(file_path)

    cols_to_keep, cols_to_drop, is_valid = inspect_sidecar_schema(file_path)

    if not is_valid:
        return {
            "file": file_path,
            "status": "skipped_not_sidecar",
            "message": "Missing 'cluster_id' or primary keys.",
            "orig_bytes": orig_size_bytes,
            "new_bytes": orig_size_bytes,
        }

    if not cols_to_drop:
        return {
            "file": file_path,
            "status": "already_clean",
            "message": f"Already a decoupled sidecar ({len(cols_to_keep)} columns).",
            "orig_bytes": orig_size_bytes,
            "new_bytes": orig_size_bytes,
            "cols_kept": len(cols_to_keep),
        }

    if dry_run:
        return {
            "file": file_path,
            "status": "dry_run",
            "cols_kept": len(cols_to_keep),
            "cols_dropped": cols_to_drop,
            "orig_bytes": orig_size_bytes,
            "new_bytes": orig_size_bytes,
        }

    # Perform streaming prune
    temp_file = file_path + ".tmp_prune.parquet"
    pf = pq.ParquetFile(file_path)

    schema_out = pa.schema([f for f in pf.schema_arrow if f.name in cols_to_keep])

    try:
        with get_parquet_writer(temp_file, schema_out) as writer:
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=cols_to_keep)
                # Ensure columns strictly match output schema
                table = table.select(cols_to_keep)
                writer.write_table(table)

        # Optional backup
        if backup:
            bak_path = file_path + ".bak"
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.link(file_path, bak_path)

        # Preserve permissions (including read-only DVC files)
        orig_stat = os.stat(file_path)
        orig_mode = stat.S_IMODE(orig_stat.st_mode)

        # Make writable before replace if read-only
        if not (orig_mode & stat.S_IWUSR):
            try:
                os.chmod(file_path, orig_mode | stat.S_IWUSR)
            except Exception:
                pass

        os.replace(temp_file, file_path)

        # Restore original permissions
        try:
            os.chmod(file_path, orig_mode)
        except Exception:
            pass

    except Exception as e:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        raise e

    new_size_bytes = os.path.getsize(file_path)
    elapsed = time.time() - start_time

    return {
        "file": file_path,
        "status": "pruned",
        "cols_kept": len(cols_to_keep),
        "cols_dropped": cols_to_drop,
        "orig_bytes": orig_size_bytes,
        "new_bytes": new_size_bytes,
        "reclaimed_bytes": orig_size_bytes - new_size_bytes,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prunes redundant base metadata columns from clustered Parquet sidecar files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["full_pipeline_output"],
        help="One or more directory paths or clustered .parquet file paths to inspect and prune.",
    )
    parser.add_argument(
        "--dir",
        "--path",
        dest="dir_opt",
        type=str,
        default=None,
        help="Explicit directory path to prune (alternative to positional argument).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan files and display columns/savings without modifying files on disk.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak hardlink before replacing pruned parquet files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="match_all",
        help="Inspect all .parquet files regardless of whether filename contains '_clustered_'.",
    )
    args = parser.parse_args()

    target_paths = args.paths
    if args.dir_opt:
        target_paths = [args.dir_opt]

    print("=" * 80)
    print("🧹 CLUSTERED SIDECAR PRUNING UTILITY")
    print("=" * 80)
    print(f"Target Paths   : {target_paths}")
    print(f"Recursive Scan : {args.recursive}")
    print(f"Dry Run Mode   : {args.dry_run}")
    print(f"Create Backups : {args.backup}")
    print("-" * 80)

    files = find_clustered_parquet_files(
        target_paths, recursive=args.recursive, match_all_parquets=args.match_all
    )

    if not files:
        print("No matching clustered .parquet files found.")
        sys.exit(0)

    print(f"Found {len(files)} candidate clustered Parquet file(s) to inspect.\n")

    total_orig_bytes = 0
    total_new_bytes = 0
    pruned_count = 0
    skipped_count = 0

    for idx, fpath in enumerate(files):
        rel_name = os.path.relpath(fpath) if os.path.isabs(fpath) else fpath
        print(f"[{idx + 1}/{len(files)}] Inspecting: {rel_name}")

        try:
            res = prune_single_sidecar(fpath, dry_run=args.dry_run, backup=args.backup)
            status = res["status"]

            if status == "already_clean":
                orig_mb = res["orig_bytes"] / (1024 * 1024)
                print(f"  -> [CLEAN] {res['message']} ({orig_mb:.1f} MB)")
                skipped_count += 1
                total_orig_bytes += res["orig_bytes"]
                total_new_bytes += res["orig_bytes"]

            elif status == "skipped_not_sidecar":
                print(f"  -> [SKIPPED] Not a clustered sidecar ({res['message']})")
                skipped_count += 1

            elif status == "dry_run":
                orig_mb = res["orig_bytes"] / (1024 * 1024)
                dropped = res["cols_dropped"]
                print(
                    f"  -> [DRY RUN] Bloated sidecar detected! Current size: {orig_mb:.1f} MB."
                )
                print(f"     Will retain {res['cols_kept']} sidecar column(s).")
                print(f"     Will drop {len(dropped)} redundant column(s): {dropped}")
                pruned_count += 1
                total_orig_bytes += res["orig_bytes"]

            elif status == "pruned":
                orig_mb = res["orig_bytes"] / (1024 * 1024)
                new_mb = res["new_bytes"] / (1024 * 1024)
                reclaimed_mb = res["reclaimed_bytes"] / (1024 * 1024)
                reclaimed_pct = (
                    (res["reclaimed_bytes"] / res["orig_bytes"]) * 100
                    if res["orig_bytes"] > 0
                    else 0
                )
                dropped = res["cols_dropped"]

                print(f"  -> ✅ Pruned {len(dropped)} redundant column(s): {dropped}")
                print(
                    f"     Size: {orig_mb:.1f} MB -> {new_mb:.1f} MB "
                    f"(-{reclaimed_mb:.1f} MB, {reclaimed_pct:.1f}% reduction in {res['elapsed']:.2f}s)"
                )
                pruned_count += 1
                total_orig_bytes += res["orig_bytes"]
                total_new_bytes += res["new_bytes"]

        except Exception as e:
            print(f"  -> [ERROR] Failed to process {fpath}: {e}", file=sys.stderr)

    print("\n" + "=" * 80)
    print("📊 PRUNING SUMMARY")
    print("=" * 80)
    print(f"Files Inspected           : {len(files)}")
    print(f"Clean Files Skipped       : {skipped_count}")
    print(f"Files Pruned (or detected): {pruned_count}")

    if not args.dry_run and pruned_count > 0:
        total_orig_mb = total_orig_bytes / (1024 * 1024)
        total_new_mb = total_new_bytes / (1024 * 1024)
        reclaimed_mb = (total_orig_bytes - total_new_bytes) / (1024 * 1024)
        total_pct = (reclaimed_mb / total_orig_mb) * 100 if total_orig_mb > 0 else 0
        print(f"Original Total Size       : {total_orig_mb:,.1f} MB")
        print(f"Pruned Total Size         : {total_new_mb:,.1f} MB")
        print(
            f"Total Reclaimed Space     : {reclaimed_mb:,.1f} MB ({total_pct:.1f}% reduction)"
        )
    elif args.dry_run and pruned_count > 0:
        total_orig_mb = total_orig_bytes / (1024 * 1024)
        print(
            f"Bloated Files Total Size  : {total_orig_mb:,.1f} MB (Run without --dry_run to prune)"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
