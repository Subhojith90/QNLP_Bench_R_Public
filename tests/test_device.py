from qnlpbench_r.device import get_device_info, select_device


def test_device_cpu_selection():
    device = select_device("cpu")
    assert str(device) == "cpu"
    assert get_device_info(device).device == "cpu"
