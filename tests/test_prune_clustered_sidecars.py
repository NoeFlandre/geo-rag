import os
import shutil
import stat
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from src.utils.prune_clustered_sidecars import (
    find_clustered_parquet_files,
    inspect_sidecar_schema,
    prune_single_sidecar,
)


class TestPruneClusteredSidecars(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_prune_bloated_sidecar(self):
        # Create a bloated sidecar parquet with base metadata + cluster columns
        data = {
            "photo_key": ["p1", "p2", "p3"],
            "Platform": ["flickr", "flickr", "osm"],
            "Photo_ID": ["101", "102", "103"],
            "cluster_id": [0, 1, 0],
            "parent_cluster_id": [0, 0, 0],
            "cluster_label": ["Forest", "Beach", "Forest"],
            "cluster_description": ["Dense trees", "Sandy coast", "Dense trees"],
            "visual_description": ["Trees", "Ocean", "Trees"],
            # Redundant base columns
            "Latitude": [48.85, 43.29, 45.76],
            "Longitude": [2.35, 5.37, 4.83],
            "Image_URL": ["http://a.jpg", "http://b.jpg", "http://c.jpg"],
            "Season": ["Summer", "Winter", "Spring"],
            "country": ["France", "France", "France"],
            "License": ["CC-BY", "CC-BY", "CC-0"],
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(self.test_dir, "dataset_model_clustered_k_2.parquet")
        df.to_parquet(file_path, compression="zstd")

        # 1. Test inspection
        cols_to_keep, cols_to_drop, is_valid = inspect_sidecar_schema(file_path)
        self.assertTrue(is_valid)
        self.assertIn("cluster_id", cols_to_keep)
        self.assertIn("cluster_label", cols_to_keep)
        self.assertIn("Latitude", cols_to_drop)
        self.assertIn("Image_URL", cols_to_drop)

        # 2. Test Dry Run
        dry_res = prune_single_sidecar(file_path, dry_run=True)
        self.assertEqual(dry_res["status"], "dry_run")
        # File should remain unpruned
        schema_after_dry = pq.ParquetFile(file_path).schema_arrow.names
        self.assertIn("Latitude", schema_after_dry)

        # 3. Test Actual Prune
        res = prune_single_sidecar(file_path, dry_run=False)
        self.assertEqual(res["status"], "pruned")
        self.assertGreater(res["reclaimed_bytes"], 0)

        # Verify pruned schema and content
        df_pruned = pd.read_parquet(file_path)
        self.assertListEqual(
            list(df_pruned.columns),
            [
                "photo_key",
                "Platform",
                "Photo_ID",
                "cluster_id",
                "parent_cluster_id",
                "cluster_label",
                "cluster_description",
                "visual_description",
            ],
        )
        self.assertNotIn("Latitude", df_pruned.columns)
        self.assertNotIn("Image_URL", df_pruned.columns)
        self.assertEqual(len(df_pruned), 3)
        self.assertEqual(df_pruned.iloc[0]["cluster_label"], "Forest")

        # 4. Test Second Run is Already Clean
        clean_res = prune_single_sidecar(file_path, dry_run=False)
        self.assertEqual(clean_res["status"], "already_clean")

    def test_skip_non_sidecar_parquet(self):
        # A file without cluster_id
        data = {"Photo_ID": ["1", "2"], "Platform": ["flickr", "flickr"]}
        df = pd.DataFrame(data)
        file_path = os.path.join(self.test_dir, "non_sidecar.parquet")
        df.to_parquet(file_path)

        res = prune_single_sidecar(file_path, dry_run=False)
        self.assertEqual(res["status"], "skipped_not_sidecar")

    def test_find_clustered_parquet_files(self):
        # Setup files in subdirectories
        sub_dir = os.path.join(self.test_dir, "nested")
        os.makedirs(sub_dir)

        p1 = os.path.join(self.test_dir, "a_clustered_k_100.parquet")
        p2 = os.path.join(self_dir_nested := sub_dir, "b_clustered_k_50.parquet")
        p3 = os.path.join(self.test_dir, "regular.parquet")
        p4 = os.path.join(self.test_dir, "a_clustered_k_100.keys.parquet")

        for p in [p1, p2, p3, p4]:
            pd.DataFrame({"x": [1]}).to_parquet(p)

        # Non-recursive
        found = find_clustered_parquet_files([self.test_dir], recursive=False)
        self.assertEqual(found, [os.path.abspath(p1)])

        # Recursive
        found_rec = find_clustered_parquet_files([self.test_dir], recursive=True)
        self.assertEqual(found_rec, sorted([os.path.abspath(p1), os.path.abspath(p2)]))

    def test_read_only_dvc_file_handling(self):
        # Create file and mark as read-only (like DVC cache)
        data = {
            "Photo_ID": ["1"],
            "Platform": ["flickr"],
            "cluster_id": [0],
            "Latitude": [10.0],
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(self.test_dir, "ro_clustered_k_1.parquet")
        df.to_parquet(file_path)
        os.chmod(file_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o444

        res = prune_single_sidecar(file_path, dry_run=False)
        self.assertEqual(res["status"], "pruned")

        df_pruned = pd.read_parquet(file_path)
        self.assertNotIn("Latitude", df_pruned.columns)
        self.assertIn("cluster_id", df_pruned.columns)


if __name__ == "__main__":
    unittest.main()
