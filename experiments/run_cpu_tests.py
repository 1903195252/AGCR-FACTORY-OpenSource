"""Run retained CPU contracts without downloading models or starting GPU jobs."""
import sys
import unittest

MODULES = ['experiments.base48.test_async_writer', 'experiments.base48.test_inference_cache', 'experiments.base48.test_model', 'experiments.base48.test_sample', 'experiments.base48.test_train', 'experiments.downstream_tracking.test_floor_proxy_association', 'experiments.downstream_tracking.test_missing_geometry', 'experiments.downstream_tracking.test_predicted_box_inference', 'experiments.downstream_tracking.test_tracking', 'experiments.formal_evaluation.test_clean_reference', 'experiments.formal_evaluation.test_condition_adherence', 'experiments.formal_evaluation.test_metrics', 'experiments.formal_evaluation.test_perturbations', 'experiments.formal_evaluation.test_predicted_box_mask', 'experiments.formal_evaluation.test_ranked_boxes', 'experiments.formal_evaluation.test_readout_calibration', 'experiments.formal_evaluation.test_state_density', 'experiments.paired_interventions.test_interventions', 'experiments.paired_interventions.test_training_contracts', 'experiments.paired_interventions.test_validation_contracts', 'experiments.paper_experiments.tests.test_stage2', 'experiments.paper_experiments.tests.test_stage2_execution', 'experiments.research_extensions.test_concat_gru_evidence', 'experiments.research_extensions.test_evidence_repeats', 'experiments.research_extensions.test_future_no_evidence']


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
