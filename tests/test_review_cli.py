"""Input/provenance checks that prevent an invalid scientific demonstration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_model", ROOT / "scripts/run_model.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
sys.path.insert(0, str(ROOT))


class ReviewCLIProvenanceTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(checkpoint_split_seed=None, normalization_seed=None, split_seed=1234)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_checkpoint_split_cannot_be_mislabelled(self):
        with self.assertRaisesRegex(ValueError, "Checkpoint split is 1234"):
            cli.checkpoint_provenance({"split_seed": 1234}, "external", self.args(split_seed=2026))

    def test_legacy_external_checkpoint_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, "no split metadata"):
            cli.checkpoint_provenance({"epoch": 290}, "unknown", self.args())
        with self.assertRaisesRegex(ValueError, "no training seed"):
            cli.checkpoint_provenance({"epoch": 290}, "unknown", self.args(checkpoint_split_seed=1234))

    def test_checkpoint_has_no_implicit_training_seed(self):
        with self.assertRaisesRegex(ValueError, "no training seed"):
            cli.checkpoint_provenance({"split_seed": 1234}, "external", self.args())
        self.assertEqual(
            cli.checkpoint_provenance({"split_seed": 1234}, "external", self.args(),
                                     require_normalization_seed=False),
            (1234, None),
        )

    def test_metadata_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            cli.checkpoint_provenance({"seed": 2026, "split_seed": 2026}, "unknown",
                                     self.args(checkpoint_split_seed=1234))

    def test_separate_normalization_seed_is_required_to_match_training(self):
        with self.assertRaisesRegex(ValueError, "normalization-seed disagrees"):
            cli.checkpoint_provenance({"seed": 1234, "split_seed": 1234}, "external",
                                     self.args(normalization_seed=2026))

    def test_documented_external_checkpoint_is_accepted(self):
        result = cli.checkpoint_provenance({"epoch": 290}, "unknown",
                                          self.args(split_seed=2026, checkpoint_split_seed=2026,
                                                    normalization_seed=2026))
        self.assertEqual(result, (2026, 2026))


class ExportMaskAlignmentTests(unittest.TestCase):
    def test_source_indices_are_rebased_to_export_mask_rows(self):
        import torch
        from train_val_test import build_export_loader
        # A source dataset/subset may carry indices into its larger parent.
        source = [(torch.tensor([11]), 91), (torch.tensor([22]), 17), (torch.tensor([33]), 62)]
        mask = torch.tensor([[True, False], [False, True], [True, True]])
        rows = []
        for labels, indices in build_export_loader(source, mask, batch_size=2):
            rows.extend(zip(labels[:, 0].tolist(), mask[indices].tolist()))
        self.assertEqual(rows, [(11, [True, False]), (22, [False, True]), (33, [True, True])])

    def test_second_oversampling_cannot_use_shorter_mask(self):
        import torch
        from train_val_test import build_export_loader
        with self.assertRaisesRegex(ValueError, 'same number of rows'):
            build_export_loader([(torch.tensor([1]), 0)] * 5,
                                torch.ones((3, 2), dtype=torch.bool), batch_size=2)


class SavedNormalizationTests(unittest.TestCase):
    def fixture(self, root, category=0):
        import numpy as np
        raw = np.zeros((10, 528, 10, 5), dtype=np.float32)
        raw[..., 1] = category
        raw[..., 2] = np.repeat(np.arange(11), 48)[None, :, None]
        raw[..., 3] = np.tile(np.arange(48), 11)[None, :, None]
        raw[..., 4] = np.arange(10)[None, None, :] + np.arange(10)[:, None, None]
        mask = np.ones((10, 528), dtype=np.uint8)
        mask[:, 0] = 0
        np.savez(root / "spm.npz", padded=raw, mask=mask)
        keys = np.array([(0, r, v) for r in range(11) for v in range(48)])
        np.savez(root / "normalization.npz", keys=keys,
                 train_means=np.full(528, 3.0), train_stds=np.full(528, 2.0))
        return raw, mask

    def test_saved_stats_round_trip_without_fitting_and_preserve_masks(self):
        import numpy as np
        import torch
        from model_files.dataset import CleanDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, mask = self.fixture(root)
            with patch.object(CleanDataset, "fit_norm_stats_from_raw", side_effect=AssertionError("must not refit")):
                clean, count, indices, coverage = cli.load_saved_demo_data(
                    root, root / "normalization.npz", 1234, 1)
            inverse, _, _ = clean.build_denorm_bundle("cpu")
            recovered = inverse(torch.tensor(clean.test_data)).numpy()
            np.testing.assert_array_equal(recovered, raw[indices])
            np.testing.assert_array_equal(clean.test_mask, mask[indices])
            self.assertEqual(count, 10)
            self.assertEqual(coverage.sum(), 528)

    def test_missing_category_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, category=1)
            with self.assertRaisesRegex(ValueError, "complete saved normalization coverage"):
                cli.load_saved_demo_data(root, root / "normalization.npz", 1234, 1)


if __name__ == "__main__":
    unittest.main()
