def test_import_package():
    import qnlpbench_r
    assert qnlpbench_r.__version__


def test_import_core_modules():
    import qnlpbench_r.config
    import qnlpbench_r.data.datasets
    import qnlpbench_r.models.baseline_models
    import qnlpbench_r.evaluation.metrics
