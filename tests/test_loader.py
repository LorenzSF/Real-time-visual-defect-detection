from real_time_visual_defect_detection.io.dataset_loader import resolve_dataset


def test_resolve_dataset_invalid_type():
    try:
        resolve_dataset("unknown", "x", "y")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
