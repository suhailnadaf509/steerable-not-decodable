"""
Tests for data generation.
Updated for EXPERIMENT_REDESIGN_SPEC.md:
  - Template IDs are T1-T8 (not A/B/C/D)
  - 8 templates per task
  - 7 OOD targets per template (not 3)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fv_cross_template.config import ExperimentConfig
from fv_cross_template.data import generate_all_prompts, validate_prompts, generate_prompts_for_task
from fv_cross_template.tasks import get_task, get_tasks


class TestPromptGeneration:
    """Test prompt generation."""

    def setup_method(self):
        self.config = ExperimentConfig(task_names=["antonym", "past_tense"])

    def test_generates_prompts(self):
        prompts = generate_all_prompts(self.config, task_names=["antonym"])
        assert "antonym" in prompts
        assert len(prompts["antonym"]) == 8  # 8 templates (T1-T8)

    def test_prompt_counts(self):
        prompts = generate_all_prompts(self.config, task_names=["antonym"])
        counts = validate_prompts(prompts)
        assert counts["icl_pos"] > 0
        assert counts["icl_neg"] > 0
        assert counts["iid"] > 0
        assert counts["ood"] > 0
        assert counts["total"] > 0

    def test_prompt_structure(self):
        prompts = generate_all_prompts(self.config, task_names=["antonym"])
        pset = prompts["antonym"]["T1"]

        # ICL positive should have demonstrations
        assert len(pset.icl_positive) > 0
        first_pos = pset.icl_positive[0]
        assert "prompt" in first_pos
        assert "expected" in first_pos
        assert "input" in first_pos
        assert "\n" in first_pos["prompt"]  # has demo lines

        # ICL negative should be bare prompts
        assert len(pset.icl_negative) > 0
        first_neg = pset.icl_negative[0]
        assert "\n" not in first_neg["prompt"]  # no demos

    def test_ood_uses_other_templates(self):
        prompts = generate_all_prompts(self.config, task_names=["antonym"])
        pset = prompts["antonym"]["T1"]

        # OOD should not include template T1
        assert "T1" not in pset.ood_test
        assert len(pset.ood_test) == 7  # T2 through T8

    def test_iid_test_has_ground_truth(self):
        task = get_task("antonym")
        prompts = generate_all_prompts(self.config, task_names=["antonym"])
        pset = prompts["antonym"]["T1"]

        for item in pset.iid_test:
            gt = task.get_ground_truth(item["input"])
            assert gt is not None, f"No ground truth for {item['input']}"
            assert gt == item["expected"]

    def test_multiple_tasks(self):
        prompts = generate_all_prompts(self.config)
        assert "antonym" in prompts
        assert "past_tense" in prompts

    def test_new_tasks_generate_prompts(self):
        """Test that new tasks (reverse_word, object_color) generate correctly."""
        config = ExperimentConfig(task_names=["reverse_word", "object_color"])
        prompts = generate_all_prompts(config)
        assert "reverse_word" in prompts
        assert "object_color" in prompts
        assert len(prompts["reverse_word"]) == 8
        assert len(prompts["object_color"]) == 8


class TestPromptDeterminism:
    """Test that prompt generation is deterministic."""

    def test_same_seed_same_output(self):
        config1 = ExperimentConfig(seed=42, task_names=["antonym"])
        config2 = ExperimentConfig(seed=42, task_names=["antonym"])

        prompts1 = generate_all_prompts(config1, task_names=["antonym"])
        prompts2 = generate_all_prompts(config2, task_names=["antonym"])

        for tid in prompts1["antonym"]:
            p1 = prompts1["antonym"][tid]
            p2 = prompts2["antonym"][tid]
            assert len(p1.icl_positive) == len(p2.icl_positive)
            for a, b in zip(p1.icl_positive, p2.icl_positive):
                assert a["input"] == b["input"]

    def test_different_seed_different_output(self):
        config1 = ExperimentConfig(seed=42, task_names=["antonym"])
        config2 = ExperimentConfig(seed=123, task_names=["antonym"])

        prompts1 = generate_all_prompts(config1, task_names=["antonym"])
        prompts2 = generate_all_prompts(config2, task_names=["antonym"])

        # At least some inputs should differ
        p1_inputs = [p["input"] for p in prompts1["antonym"]["T1"].icl_positive]
        p2_inputs = [p["input"] for p in prompts2["antonym"]["T1"].icl_positive]
        assert p1_inputs != p2_inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
