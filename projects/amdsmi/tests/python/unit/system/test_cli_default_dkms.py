#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bare amd-smi header DKMS row tests."""

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class _LibraryError(Exception):
    def get_error_info(self):
        return "not supported"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _project_root() / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raise(*_args, **_kwargs):
    raise _LibraryError()


def _load_default_module(dkms_getter):
    interface = types.SimpleNamespace(
        amdsmi_get_rocm_version=lambda: (True, "10.1.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _gpu: {"driver_version": "6.19.14.31400000"},
        amdsmi_get_amdgpu_dkms_version=dkms_getter,
        amdsmi_get_fw_info=_raise,
        amdsmi_get_gpu_vbios_info=_raise,
        amdsmi_get_gpu_metrics_info=_raise,
        amdsmi_get_gpu_memory_partition=_raise,
        amdsmi_get_gpu_accelerator_partition_profile=_raise,
        amdsmi_get_gpu_asic_info=_raise,
        amdsmi_get_gpu_device_bdf=_raise,
        amdsmi_get_gpu_enumeration_info=_raise,
        amdsmi_get_power_cap_info=_raise,
        amdsmi_get_gpu_memory_usage=_raise,
        amdsmi_get_gpu_memory_total=_raise,
        amdsmi_get_gpu_total_ecc_count=_raise,
        amdsmi_get_gpu_fan_speed=_raise,
        amdsmi_get_gpu_fan_speed_max=_raise,
        amdsmi_get_gpu_process_list=_raise,
        _NA_amdsmi_get_gpu_metrics_info=lambda: "N/A",
    )
    amdsmi = types.ModuleType("amdsmi")
    amdsmi.amdsmi_interface = interface
    amdsmi.amdsmi_exception = types.SimpleNamespace(AmdSmiLibraryException=_LibraryError)
    version_metadata = types.ModuleType("_version")
    version_metadata.__version__ = "1.0.0"
    helpers_module = types.ModuleType("amdsmi_helpers")
    helpers_module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})

    old_modules = {name: sys.modules.get(name) for name in ("amdsmi", "_version", "amdsmi_helpers")}
    sys.modules["amdsmi"] = amdsmi
    sys.modules["_version"] = version_metadata
    sys.modules["amdsmi_helpers"] = helpers_module
    try:
        return _load_module("amdsmi_cli/subcommands/default.py", "default_dkms_under_test")
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _run_default(dkms_getter):
    module = _load_default_module(dkms_getter)
    commands = object.__new__(module.DefaultCommands)
    captured = {}

    class _Logger:
        def is_json_format(self):
            return True

        def is_csv_format(self):
            return False

        def print_output(self):
            return None

    commands.logger = _Logger()
    commands.helpers = types.SimpleNamespace(
        is_amdgpu_initialized=lambda: True,
        check_required_groups=lambda: None,
        get_gpu_id_from_device_handle=lambda _gpu: 0,
        get_apu_memory_type_and_name=lambda _gpu, _gpu_id: (0, "VRAM"),
        convert_SI_unit=lambda value, _unit: value,
        convert_bytes_to_readable=lambda value: str(value),
        unit_format=lambda _logger, value, _unit: str(value),
    )
    commands.group_check_printed = True
    commands.default(types.SimpleNamespace())
    captured["output"] = commands.logger.output
    return captured["output"]


def _load_logger_module():
    helpers_module = types.ModuleType("amdsmi_helpers")
    helpers_module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    old_helpers = sys.modules.get("amdsmi_helpers")
    sys.modules["amdsmi_helpers"] = helpers_module
    try:
        return _load_module("amdsmi_cli/amdsmi_logger.py", "logger_dkms_under_test")
    finally:
        if old_helpers is None:
            sys.modules.pop("amdsmi_helpers", None)
        else:
            sys.modules["amdsmi_helpers"] = old_helpers


def _banner_payload(dkms_version):
    return {
        "version_info": {
            "amd-smi": "27.1.0",
            "amdgpu version": {"driver_version": "6.19.14.31400000"},
            "amdgpu dkms version": dkms_version,
            "kernel version": "6.8.0-124-generic",
            "fw pldm version": "N/A",
            "vbios version": "N/A",
            "rocm version": (True, "10.1.0"),
        },
        "gpu_info_list": [],
        "processes": [],
    }


class TestDefaultDkmsHeader(unittest.TestCase):
    def test_default_payload_includes_active_dkms_version(self):
        output = _run_default(lambda: "6.19.14-2370381.24.04")
        self.assertEqual(output["version_info"]["amdgpu dkms version"], "6.19.14-2370381.24.04")

    def test_default_payload_uses_na_when_dkms_is_unsupported(self):
        def _unsupported():
            raise _LibraryError()

        output = _run_default(_unsupported)
        self.assertEqual(output["version_info"]["amdgpu dkms version"], "N/A")

    def test_banner_prints_dkms_row_after_amdgpu_version(self):
        module = _load_logger_module()
        logger = module.AMDSMILogger(
            helpers=types.SimpleNamespace(os_info=lambda: "Linux Baremetal")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            logger.print_default_output(_banner_payload("6.19.14-2370381.24.04"))
        lines = stdout.getvalue().splitlines()
        amdgpu_index = lines.index(
            next(line for line in lines if line.startswith("| amdgpu Version:"))
        )
        self.assertTrue(lines[amdgpu_index + 1].startswith("| amdgpu DKMS:"))
        self.assertIn("6.19.14-2370381.24.04", lines[amdgpu_index + 1])

    def test_banner_omits_dkms_row_when_version_is_na(self):
        module = _load_logger_module()
        logger = module.AMDSMILogger(
            helpers=types.SimpleNamespace(os_info=lambda: "Linux Baremetal")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            logger.print_default_output(_banner_payload("N/A"))
        self.assertFalse(
            any(line.startswith("| amdgpu DKMS:") for line in stdout.getvalue().splitlines())
        )


if __name__ == "__main__":
    unittest.main()
