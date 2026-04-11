"""
Tests for task definitions and accuracy evaluation.
Updated for EXPERIMENT_REDESIGN_SPEC.md:
  - 12 tasks, 8 templates each (T1-T8)
  - 5 categories (LEXICAL_RETRIEVAL, FACTUAL_RETRIEVAL, etc.)
  - No present_to_gerund or singular_past
  - New tasks: reverse_word, object_color
  - All templates task-specific (no shared "{X} ->")
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fv_cross_template.tasks import (
    TASK_REGISTRY, TaskSpec, AccuracyMode, TaskCategory, TemplateStyle,
    TEMPLATE_STYLE_MAP,
    get_task, get_tasks, validate_all_tasks, build_task_registry,
)


class TestTaskRegistry:
    """Test that the task registry is well-formed per spec."""

    def test_registry_has_exactly_12_tasks(self):
        assert len(TASK_REGISTRY) == 12, f"Expected 12 tasks, got {len(TASK_REGISTRY)}"

    def test_expected_task_names(self):
        expected = {
            "antonym", "synonym", "hypernym",
            "country_capital", "english_spanish", "object_color",
            "past_tense", "plural",
            "capitalize", "first_letter", "reverse_word",
            "sentiment_flip",
        }
        assert set(TASK_REGISTRY.keys()) == expected

    def test_removed_tasks_absent(self):
        assert "present_to_gerund" not in TASK_REGISTRY
        assert "singular_past" not in TASK_REGISTRY

    def test_all_tasks_have_8_templates(self):
        for name, spec in TASK_REGISTRY.items():
            assert spec.n_templates == 8, f"Task '{name}' has {spec.n_templates} templates (need 8)"

    def test_template_ids_are_T1_through_T8(self):
        expected_ids = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
        for name, spec in TASK_REGISTRY.items():
            assert spec.template_ids == expected_ids, (
                f"Task '{name}' has template IDs {spec.template_ids}, expected {expected_ids}"
            )

    def test_all_tasks_have_enough_pairs(self):
        for name, spec in TASK_REGISTRY.items():
            assert spec.n_pairs >= 60, (
                f"Task '{name}' has only {spec.n_pairs} pairs (need >= 60)"
            )

    def test_specific_pair_counts(self):
        """Verify pair counts match spec Section 2.5."""
        expected = {
            "antonym": 95, "synonym": 88, "hypernym": 86,
            "country_capital": 90, "english_spanish": 88, "object_color": 85,
            "past_tense": 90, "plural": 90,
            "capitalize": 84, "first_letter": 86,
            "reverse_word": 80, "sentiment_flip": 60,
        }
        for name, count in expected.items():
            actual = TASK_REGISTRY[name].n_pairs
            assert actual == count, (
                f"Task '{name}': expected {count} pairs, got {actual}"
            )

    def test_all_5_categories_represented(self):
        categories = {spec.category for spec in TASK_REGISTRY.values()}
        assert TaskCategory.LEXICAL_RETRIEVAL in categories
        assert TaskCategory.FACTUAL_RETRIEVAL in categories
        assert TaskCategory.MORPHOLOGICAL_TRANSFORM in categories
        assert TaskCategory.CHARACTER_SURFACE in categories
        assert TaskCategory.COMPOSITIONAL_SEMANTIC in categories

    def test_category_assignments(self):
        """Verify category assignments match spec Section 2.5."""
        assert TASK_REGISTRY["antonym"].category == TaskCategory.LEXICAL_RETRIEVAL
        assert TASK_REGISTRY["synonym"].category == TaskCategory.LEXICAL_RETRIEVAL
        assert TASK_REGISTRY["hypernym"].category == TaskCategory.LEXICAL_RETRIEVAL
        assert TASK_REGISTRY["country_capital"].category == TaskCategory.FACTUAL_RETRIEVAL
        assert TASK_REGISTRY["english_spanish"].category == TaskCategory.FACTUAL_RETRIEVAL
        assert TASK_REGISTRY["object_color"].category == TaskCategory.FACTUAL_RETRIEVAL
        assert TASK_REGISTRY["past_tense"].category == TaskCategory.MORPHOLOGICAL_TRANSFORM
        assert TASK_REGISTRY["plural"].category == TaskCategory.MORPHOLOGICAL_TRANSFORM
        assert TASK_REGISTRY["capitalize"].category == TaskCategory.CHARACTER_SURFACE
        assert TASK_REGISTRY["first_letter"].category == TaskCategory.CHARACTER_SURFACE
        assert TASK_REGISTRY["reverse_word"].category == TaskCategory.CHARACTER_SURFACE
        assert TASK_REGISTRY["sentiment_flip"].category == TaskCategory.COMPOSITIONAL_SEMANTIC

    def test_no_shared_templates_across_tasks(self):
        """Verify no two tasks share an identical template string (spec §3.5)."""
        all_templates = []
        for name, spec in TASK_REGISTRY.items():
            for tid, tstr in spec.templates.items():
                all_templates.append((name, tid, tstr))

        # Check for duplicates
        seen = {}
        for name, tid, tstr in all_templates:
            if tstr in seen:
                pytest.fail(
                    f"Template string shared between '{seen[tstr]}' and "
                    f"'{name}/{tid}': {tstr}"
                )
            seen[tstr] = f"{name}/{tid}"

    def test_no_task_ambiguous_arrow_template(self):
        """No task should use bare '{X} ->' (spec §3.1 point 4)."""
        for name, spec in TASK_REGISTRY.items():
            for tid, tstr in spec.templates.items():
                assert tstr.strip() != "{X} ->", (
                    f"Task '{name}' template {tid} uses forbidden '{'{X}'} ->' template"
                )

    def test_all_templates_contain_placeholder(self):
        for name, spec in TASK_REGISTRY.items():
            for tid, tstr in spec.templates.items():
                assert "{X}" in tstr, (
                    f"Task '{name}' template {tid} missing {{X}} placeholder"
                )

    def test_template_style_map(self):
        assert TEMPLATE_STYLE_MAP["T1"] == TemplateStyle.NATURAL
        assert TEMPLATE_STYLE_MAP["T2"] == TemplateStyle.NATURAL
        assert TEMPLATE_STYLE_MAP["T3"] == TemplateStyle.SYMBOLIC
        assert TEMPLATE_STYLE_MAP["T4"] == TemplateStyle.SYMBOLIC
        assert TEMPLATE_STYLE_MAP["T5"] == TemplateStyle.QUESTION
        assert TEMPLATE_STYLE_MAP["T6"] == TemplateStyle.QUESTION
        assert TEMPLATE_STYLE_MAP["T7"] == TemplateStyle.FORMAL
        assert TEMPLATE_STYLE_MAP["T8"] == TemplateStyle.FORMAL

    def test_get_task_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task_xyz")

    def test_get_tasks_all(self):
        all_tasks = get_tasks()
        assert len(all_tasks) == len(TASK_REGISTRY)

    def test_get_tasks_subset(self):
        subset = get_tasks(["antonym", "synonym"])
        assert len(subset) == 2
        assert "antonym" in subset
        assert "synonym" in subset

    def test_no_duplicate_inputs(self):
        for name, spec in TASK_REGISTRY.items():
            inputs = [inp for inp, _ in spec.pairs]
            assert len(inputs) == len(set(inputs)), (
                f"Task '{name}' has duplicate inputs"
            )


class TestMaxNewTokens:
    """Test task-specific max_new_tokens per spec Section 2.2."""

    def test_first_letter_max_tokens(self):
        assert get_task("first_letter").max_new_tokens == 3

    def test_sentiment_flip_max_tokens(self):
        assert get_task("sentiment_flip").max_new_tokens == 10

    def test_default_max_tokens(self):
        for name in ["antonym", "synonym", "country_capital", "past_tense",
                      "plural", "capitalize", "reverse_word", "object_color"]:
            assert get_task(name).max_new_tokens == 5


class TestAccuracyEvaluation:
    """Test task-specific accuracy evaluation."""

    def test_antonym_substring(self):
        spec = get_task("antonym")
        assert spec.check_accuracy("hot", " cold and icy")
        assert spec.check_accuracy("hot", "cold")
        assert not spec.check_accuracy("hot", "warm")
        assert not spec.check_accuracy("hot", "")

    def test_synonym_substring_with_alternatives(self):
        spec = get_task("synonym")
        # Primary output
        assert spec.check_accuracy("happy", "glad")
        # Alternative valid outputs
        assert spec.check_accuracy("happy", "joyful")
        assert spec.check_accuracy("happy", "cheerful")
        assert not spec.check_accuracy("happy", "sad")

    def test_country_capital(self):
        spec = get_task("country_capital")
        assert spec.check_accuracy("France", "Paris is the capital")
        assert spec.check_accuracy("France", "Paris")
        assert not spec.check_accuracy("France", "London")

    def test_past_tense(self):
        spec = get_task("past_tense")
        assert spec.check_accuracy("walk", "walked")
        assert spec.check_accuracy("walk", " walked across")
        assert not spec.check_accuracy("walk", "walking")

    def test_plural(self):
        spec = get_task("plural")
        assert spec.check_accuracy("cat", "cats")
        assert not spec.check_accuracy("cat", "cat")

    def test_capitalize_case_sensitive(self):
        spec = get_task("capitalize")
        assert spec.check_accuracy("hello", "HELLO")
        assert spec.check_accuracy("hello", " HELLO world")
        assert not spec.check_accuracy("hello", "hello")
        assert not spec.check_accuracy("hello", "Hello")

    def test_first_letter(self):
        spec = get_task("first_letter")
        assert spec.check_accuracy("apple", "a")
        assert spec.check_accuracy("apple", " a is the first letter")
        assert not spec.check_accuracy("apple", "b")

    def test_reverse_word(self):
        spec = get_task("reverse_word")
        assert spec.check_accuracy("hello", "olleh")
        assert spec.check_accuracy("hello", " olleh!")
        assert not spec.check_accuracy("hello", "hello")

    def test_object_color(self):
        spec = get_task("object_color")
        assert spec.check_accuracy("banana", "yellow")
        assert spec.check_accuracy("banana", " yellow fruit")
        assert not spec.check_accuracy("banana", "purple")

    def test_object_color_alternatives(self):
        spec = get_task("object_color")
        # Rose has multiple valid colors
        assert spec.check_accuracy("rose", "red")
        assert spec.check_accuracy("rose", "pink")

    def test_ground_truth_lookup(self):
        spec = get_task("antonym")
        assert spec.get_ground_truth("hot") == "cold"
        assert spec.get_ground_truth("nonexistent_word") is None


class TestTaskValidation:
    """Test task validation."""

    def test_validate_all_tasks_passes(self):
        warnings = validate_all_tasks()
        total_warnings = sum(len(w) for w in warnings.values())
        assert total_warnings == 0, (
            f"Built-in tasks have {total_warnings} validation warnings: {warnings}"
        )

    def test_validate_bad_task(self):
        bad_task = TaskSpec(
            name="bad", category=TaskCategory.LEXICAL_RETRIEVAL,
            description="test", expected_difficulty="easy",
            templates={"T1": "{X} test"},
            pairs=[("a", "b"), ("c", "d")],  # too few pairs + templates
        )
        warnings = bad_task.validate()
        assert len(warnings) > 0
        assert any("pairs" in w.lower() or "60" in w for w in warnings)
        assert any("templates" in w.lower() or "8" in w for w in warnings)


class TestConfigIntegration:
    """Test config integration."""

    def test_experiment_config_defaults(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        assert config.model_key == "llama-3.1-8b-base"
        assert config.science.iid_accuracy_threshold == 0.10
        assert len(config.science.steering_strengths) >= 5

    def test_config_probe_validation_split(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        assert config.science.probe_train_fraction == 0.70
        assert config.science.probe_val_fraction == 0.15
        assert config.science.probe_test_fraction == 0.15
        total = (config.science.probe_train_fraction +
                 config.science.probe_val_fraction +
                 config.science.probe_test_fraction)
        assert abs(total - 1.0) < 1e-6

    def test_config_probe_c_sweep(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        assert config.science.probe_regularization_sweep == [0.01, 0.1, 1.0, 10.0]

    def test_config_data_derived_thresholds_default_on(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        assert config.science.use_data_derived_thresholds is True

    def test_config_stages_include_baseline(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        assert "baseline" in config.stages
        assert "probe_activations" in config.stages

    def test_config_validation(self):
        from fv_cross_template.config import ExperimentConfig
        config = ExperimentConfig()
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_config_bad_model(self):
        from fv_cross_template.config import ExperimentConfig
        with pytest.raises(ValueError, match="Unknown model key"):
            ExperimentConfig(model_key="nonexistent-model")

    def test_model_spec_layers(self):
        from fv_cross_template.config import MODEL_REGISTRY
        spec = MODEL_REGISTRY["llama-3.1-8b-base"]
        layers = spec.extraction_layers()
        assert layers[0] == 2
        assert layers[-1] == 32
        assert len(layers) == 16  # every 2nd layer of 32

    def test_gemma_spec_layers(self):
        from fv_cross_template.config import MODEL_REGISTRY
        spec = MODEL_REGISTRY["gemma-2-9b-base"]
        layers = spec.extraction_layers()
        assert layers[0] == 2
        assert layers[-1] == 42
        assert len(layers) == 21  # every 2nd layer of 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
