from types import SimpleNamespace

import pytest

from openpi_client.franka_collect.test_supervisor import hardware_fault


@pytest.mark.parametrize(
    "name,level,stamp,expected",
    [
        ("FrankaHardwareInterface", 40, 11, True),
        ("left.controller_manager", 40, 11, True),
        ("FrankaHardwareInterface", 30, 11, False),
        ("FrankaHardwareInterface", 40, 9, False),
        ("camera.primary", 40, 11, False),
    ],
)
def test_hardware_monitor_rejects_stale_and_unrelated_errors(name, level, stamp, expected):
    message = SimpleNamespace(name=name, level=level, stamp=SimpleNamespace(sec=stamp, nanosec=0))
    assert hardware_fault(message, 10_000_000_000) == expected
