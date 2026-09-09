#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/run-vfio-guest.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "scripts" / "run-vfio-guest.py"
SPEC = importlib.util.spec_from_file_location("run_vfio_guest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def write_file(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def fake_qemu(
    path: Path,
    delay: float = 0.05,
    returncode: int = 0,
    output: str = "workload: PASS",
    supports_vfio: bool = True,
) -> None:
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import sys\n"
            "import time\n"
            "if sys.argv[1:3] == ['-accel', 'help']:\n"
            "    print('Accelerators supported in QEMU binary:')\n"
            "    print('tcg')\n"
            "elif sys.argv[1:3] == ['-device', 'help']:\n"
            f"    print('name \\\"vfio-user-pci\\\"') if {supports_vfio!r} else None\n"
            "else:\n"
            "    Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
            f"    print({output!r}, flush=True)\n"
            f"    time.sleep({delay!r})\n"
            f"    raise SystemExit({returncode})\n"
        ).encode(),
        0o755,
    )


def fake_server(path: Path, behavior: str, supports_vfio: bool = True) -> None:
    actions = {
        "clean": (
            "stopping = False\n"
            "def stop(_signal, _frame):\n"
            "    global stopping\n"
            "    stopping = True\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "while not stopping:\n"
            "    time.sleep(0.01)\n"
        ),
        "fail": "time.sleep(0.08)\nraise SystemExit(23)\n",
        "exit_clean": "time.sleep(0.15)\n",
        "fail_on_stop": (
            "def fail(_signal, _frame):\n"
            "    raise SystemExit(23)\n"
            "signal.signal(signal.SIGTERM, fail)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        ),
        "ignore": (
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        ),
    }
    if behavior == "startup_fail":
        startup = "raise SystemExit(23)\n"
        action = ""
    else:
        startup = (
            "socket_path = sys.argv[sys.argv.index('--vfio-socket') + 1]\n"
            "ready_fd = int(sys.argv[sys.argv.index('--vfio-ready-fd') + 1])\n"
            "listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "listener.bind(socket_path)\n"
            "listener.listen()\n"
            "os.write(ready_fd, b'\\x01')\n"
            "os.close(ready_fd)\n"
        )
        action = actions[behavior]
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import signal\n"
            "import socket\n"
            "import sys\n"
            "import time\n"
            "if sys.argv[1:] == ['--check-vfio-user']:\n"
            f"    if {supports_vfio!r}:\n"
            "        print('vfio-user support enabled')\n"
            "        raise SystemExit(0)\n"
            "    print('this build has no vfio-user support', file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
            + startup
            + action
        ).encode(),
        0o755,
    )


@unittest.skipUnless(sys.platform.startswith("linux"), "VFIO launcher is Linux-only")
class RunVfioGuestTest(unittest.TestCase):
    def run_fake_guest(
        self,
        root: Path,
        server_behavior: str,
        qemu_delay: float = 0.05,
        qemu_returncode: int = 0,
        qemu_output: str = "workload: PASS",
        timeout: float = 1.0,
    ) -> tuple[int, str]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu, qemu_delay, qemu_returncode, qemu_output)
        fake_server(rocjitsu, server_behavior)

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = RUNNER.main(
                [
                    "--kernel",
                    str(kernel),
                    "--initramfs",
                    str(initramfs),
                    "--qemu",
                    str(qemu),
                    "--rocjitsu",
                    str(rocjitsu),
                    "--config",
                    str(config),
                    "--output",
                    str(root / "run"),
                    "--accel",
                    "tcg",
                    "--timeout",
                    str(timeout),
                    "--expect-log",
                    "workload: PASS",
                ]
            )
        return status, diagnostics.getvalue()

    def assert_fake_processes_reaped(self, root: Path) -> None:
        for name in ("qemu", "rocjitsu"):
            self.assert_fake_process_reaped(root, name)

    def assert_fake_process_reaped(self, root: Path, name: str) -> None:
        pid_file = (root / name).with_suffix(".pid")
        self.assertTrue(pid_file.is_file(), f"{name} never recorded its pid")
        pid = int(pid_file.read_text(encoding="utf-8"))
        with self.assertRaises(ChildProcessError, msg=f"{name} was not reaped"):
            os.waitpid(pid, os.WNOHANG)

    def test_auto_acceleration_falls_back_to_tcg(self) -> None:
        self.assertEqual(
            RUNNER.select_accelerator(
                "auto", {"kvm", "tcg"}, Path("/definitely/missing/kvm")
            ),
            "tcg",
        )

    def test_timeouts_must_be_positive_and_finite(self) -> None:
        required = [
            "--kernel",
            "kernel",
            "--initramfs",
            "initramfs",
            "--qemu",
            "qemu",
            "--rocjitsu",
            "rocjitsu",
            "--config",
            "config",
            "--output",
            "output",
        ]
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    RUNNER.parse_arguments([*required, "--timeout", value])

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                RUNNER.parse_arguments([*required, "--socket-timeout", "12.5"])

    def test_qemu_arguments_preserve_caller_policy(self) -> None:
        arguments = RUNNER.build_qemu_arguments(
            Path("/qemu"),
            Path("/guest/vmlinuz"),
            Path("/guest/initramfs.gz"),
            Path("/tmp/device.sock"),
            "tcg",
            "3G",
            "console=ttyS0 rdinit=/init workload=gemm",
        )

        self.assertEqual(arguments[arguments.index("-cpu") + 1], "qemu64,+hypervisor")
        self.assertEqual(arguments[arguments.index("-m") + 1], "3G")
        self.assertIn("memory-backend-memfd,id=mem,size=3G,share=on", arguments)
        self.assertEqual(
            arguments[arguments.index("-append") + 1],
            "console=ttyS0 rdinit=/init workload=gemm",
        )
        device = json.loads(arguments[arguments.index("-device") + 1])
        self.assertEqual(device["driver"], "vfio-user-pci")
        self.assertEqual(device["rombar"], 0)
        self.assertEqual(device["socket"], "/tmp/device.sock")

    def test_log_checks_are_generic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "guest.log"
            log.write_text("kernel booted\nworkload: PASS\n", encoding="utf-8")
            RUNNER.check_guest_log(log, ["workload: PASS"], ["workload: FAIL"])
            with self.assertRaisesRegex(RUNNER.GuestRunError, "rejected text"):
                RUNNER.check_guest_log(log, [], ["kernel booted"])

    def test_dry_run_prints_commands_without_persisting_input_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            fake_qemu(qemu)
            fake_server(rocjitsu, "clean")
            output = root / "run"

            diagnostics = io.StringIO()
            commands = io.StringIO()
            with contextlib.redirect_stderr(diagnostics), contextlib.redirect_stdout(
                commands
            ):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--append",
                        "console=ttyS0 test=1",
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0, diagnostics.getvalue())
            self.assertIn("--vfio-socket", commands.getvalue())
            self.assertNotIn("--vfio-ready-fd", commands.getvalue())
            self.assertIn("console=ttyS0 test=1", commands.getvalue())
            self.assertFalse((output / "run-manifest.json").exists())
            self.assertFalse(output.exists())

    def test_dry_run_rejects_a_non_executable_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            fake_qemu(qemu)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o644)

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("rocjitsu is not executable", diagnostics.getvalue())

    def test_dry_run_rejects_qemu_without_vfio_user_pci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            fake_qemu(qemu, supports_vfio=False)
            fake_server(rocjitsu, "clean")

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn(
                "does not provide the vfio-user-pci device", diagnostics.getvalue()
            )

    def test_dry_run_rejects_rocjitsu_without_vfio_user_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            fake_qemu(qemu)
            fake_server(rocjitsu, "clean", supports_vfio=False)

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("no usable vfio-user support", diagnostics.getvalue())

    def test_preexisting_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            write_file(qemu, b"#!/bin/sh\nexit 0\n", 0o755)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o755)
            output = root / "run"
            output.mkdir()
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("run output directory already exists", diagnostics.getvalue())

    def test_socket_must_be_owned_by_the_run_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            write_file(qemu, b"#!/bin/sh\nexit 0\n", 0o755)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o755)
            external_socket = root / "unrelated.sock"
            external_socket.write_text("owned by another service", encoding="utf-8")
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--socket",
                        str(external_socket),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("socket must be directly inside", diagnostics.getvalue())
            self.assertEqual(
                external_socket.read_text(encoding="utf-8"),
                "owned by another service",
            )

    def test_readiness_signal_observes_a_real_unix_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(socket_path))
                listener.listen()
                ready_read, ready_write = os.pipe()
                try:
                    os.write(ready_write, b"\x01")
                    identity = RUNNER.wait_for_server_ready(
                        socket_path,
                        mock.Mock(poll=mock.Mock(return_value=None)),
                        ready_read,
                    )
                finally:
                    os.close(ready_write)
                    os.close(ready_read)

            socket_stat = socket_path.lstat()
            self.assertEqual(identity, (socket_stat.st_dev, socket_stat.st_ino))

    def test_readiness_pipe_eof_reports_startup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            ready_read, ready_write = os.pipe()
            os.close(ready_write)
            try:
                with self.assertRaisesRegex(
                    RUNNER.GuestRunError, "closed its readiness channel"
                ):
                    RUNNER.wait_for_server_ready(
                        socket_path,
                        mock.Mock(poll=mock.Mock(return_value=23)),
                        ready_read,
                    )
            finally:
                os.close(ready_read)

    def test_cleanup_removes_only_the_observed_real_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            with socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as original, socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as replacement:
                original.bind(str(socket_path))
                original_identity = (
                    socket_path.lstat().st_dev,
                    socket_path.lstat().st_ino,
                )
                socket_path.unlink()
                replacement.bind(str(socket_path))
                replacement_identity = (
                    socket_path.lstat().st_dev,
                    socket_path.lstat().st_ino,
                )
                RUNNER.remove_owned_socket(socket_path, original_identity)
                self.assertTrue(socket_path.exists())
                RUNNER.remove_owned_socket(socket_path, replacement_identity)
                self.assertFalse(socket_path.exists())

    def test_server_failure_is_not_masked_by_successful_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "fail", qemu_delay=0.2)
            self.assertIn(
                "workload: PASS",
                (root / "run/guest.log").read_text(encoding="utf-8"),
            )
            self.assert_fake_processes_reaped(root)
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_server_startup_failure_does_not_start_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "startup_fail")
            self.assert_fake_process_reaped(root, "rocjitsu")
            self.assertFalse((root / "qemu.pid").exists())
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("closed its readiness channel", diagnostics)

    def test_clean_server_exit_allows_qemu_to_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "exit_clean", qemu_delay=1.1, timeout=2.0
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 0, diagnostics)

    def test_qemu_timeout_fails_and_reaps_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "clean", qemu_delay=1.0, timeout=0.05
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("QEMU did not exit within", diagnostics)

    def test_nonzero_qemu_exit_fails_and_reaps_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "clean", qemu_returncode=17)
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("QEMU exited with status 17", diagnostics)

    def test_missing_log_marker_fails_and_reaps_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "clean", qemu_output="workload finished"
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("guest log is missing expected text", diagnostics)

    def test_guest_failure_is_preserved_when_server_cleanup_also_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "fail_on_stop", qemu_returncode=17
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("QEMU exited with status 17", diagnostics)
        self.assertIn(
            "cleanup also failed: rocjitsu exited with status 23", diagnostics
        )

    def test_nonzero_server_shutdown_fails_after_qemu_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "fail_on_stop")
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_clean_server_shutdown_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "clean")
            self.assertEqual(stat.S_IMODE((root / "run").stat().st_mode), 0o700)
            self.assert_fake_processes_reaped(root)
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 0, diagnostics)

    def test_server_that_requires_forced_kill_fails_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            RUNNER, "PROCESS_SHUTDOWN_TIMEOUT", 0.1
        ):
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "ignore")
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("did not exit after SIGTERM and was killed", diagnostics)


if __name__ == "__main__":
    unittest.main()
