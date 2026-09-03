import tempfile
import unittest
from pathlib import Path

from benchmarks.phase_e_stage import (
    checkpoint_pipeline,
    load_pipeline_checkpoint,
    split_stage_stories,
)
from mrdl.language import EmbeddingTable, MRDLLanguagePipeline


class PhaseEStageToolingTests(unittest.TestCase):
    def test_split_is_whole_story_and_non_overlapping(self):
        records = [
            {"story_id": f"story-{index}", "source_index": index, "tokens": ["<bos>", str(index), "<eos>"]}
            for index in range(40)
        ]
        train, test, manifest = split_stage_stories(records, split_seed=1729)
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertFalse(set(manifest["train_story_ids"]) & set(manifest["test_story_ids"]))
        self.assertEqual(len(records), len(train) + len(test))

    def test_checkpoint_round_trip_preserves_model_counts(self):
        stories = [["<bos>", "a", "b", "<eos>"] for _ in range(3)]
        embeddings = EmbeddingTable.random_frozen({token for story in stories for token in story}, 8, 7)
        pipeline = MRDLLanguagePipeline(embeddings, beam_width=4)
        pipeline.observe_training(stories)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.checkpoint.json"
            checkpoint_pipeline(
                pipeline,
                path,
                {"completed_story_count": 3, "resume_next_source_index": 3},
            )
            restored = load_pipeline_checkpoint(path)
        self.assertEqual(len(pipeline.edge_memory.edges), len(restored.edge_memory.edges))
        self.assertEqual(pipeline.controller.weights, restored.controller.weights)
        self.assertEqual(pipeline.edge_memory.delta_updates, restored.edge_memory.delta_updates)


if __name__ == "__main__":
    unittest.main()
