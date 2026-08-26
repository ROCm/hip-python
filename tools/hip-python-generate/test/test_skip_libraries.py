# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Tests for ``--skip-libraries``: the parsing and the library filter.

The option carries a comma-separated list of library names from a ``/full
test`` comment down to the generator, through a workflow input, an environment
variable and a cmake cache entry, none of which understand the comma. The
generator is where the list is finally split, so the splitting and the
filtering it feeds are worth pinning here -- neither needs a ROCm installation
to exercise.
"""

import pytest
from hip_python_codegen.binding_generator import (
    AVAILABLE_GENERATORS,
    SKIPPED_BY_REQUEST,
    _parse_skip_libraries,
    _resolve_skip_libraries,
)
from hip_python_codegen.generate import parse_unified_args


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, set()),
        ("", set()),
        ("hiptensor", {"hiptensor"}),
        ("hiptensor,hipdnn_backend", {"hiptensor", "hipdnn_backend"}),
        # Repeatable flag: argparse hands over a list of what was passed.
        (["hiptensor", "hipdnn_backend"], {"hiptensor", "hipdnn_backend"}),
        (
            ["hiptensor,rccl", "hipdnn_backend"],
            {"hiptensor", "rccl", "hipdnn_backend"},
        ),
        # Whitespace and empty elements are tolerated on a command line, where
        # a shell user may well write the list with spaces after the commas.
        (" hiptensor , hipdnn_backend ", {"hiptensor", "hipdnn_backend"}),
        ("hiptensor,,hipdnn_backend,", {"hiptensor", "hipdnn_backend"}),
        ([",", " "], set()),
    ],
)
def test_parse_skip_libraries(value, expected):
    assert _parse_skip_libraries(value) == expected


def test_resolve_drops_the_named_libraries():
    lib_names = ["hip", "hiprtc", "hiptensor", "hipdnn_backend"]
    kept, requested = _resolve_skip_libraries(
        lib_names, "hiptensor,hipdnn_backend"
    )
    assert kept == ["hip", "hiprtc"]
    assert requested == {"hiptensor", "hipdnn_backend"}


def test_resolve_keeps_the_order_of_the_remaining_libraries():
    lib_names = list(AVAILABLE_GENERATORS)
    kept, _ = _resolve_skip_libraries(lib_names, "hiprtc")
    assert kept == [name for name in lib_names if name != "hiprtc"]


def test_resolve_ignores_a_library_outside_the_selection():
    """A wheel that was not selected has nothing to skip.

    Reporting it as skipped would put a library in ``skipped.txt`` that this
    run was never going to generate, and the CI summary reads that file.
    """
    kept, requested = _resolve_skip_libraries(["hip", "hiprtc"], "hiptensor")
    assert kept == ["hip", "hiprtc"]
    assert requested == set()


def test_resolve_reports_every_unknown_name_at_once():
    with pytest.raises(KeyError) as excinfo:
        _resolve_skip_libraries(
            list(AVAILABLE_GENERATORS), "hipTensor,nosuchlib"
        )
    message = str(excinfo.value)
    assert "hipTensor" in message
    assert "nosuchlib" in message


def test_skipped_by_request_is_a_distinct_reason():
    """CI tells a requested skip from a missing header by this string."""
    assert SKIPPED_BY_REQUEST == "skipped by request"


@pytest.mark.parametrize(
    "argv,expected",
    [
        (
            ["--skip-libraries", "hiptensor,hipdnn_backend"],
            {"hiptensor", "hipdnn_backend"},
        ),
        (
            [
                "--skip-libraries",
                "hiptensor",
                "--skip-libraries",
                "hipdnn_backend",
            ],
            {"hiptensor", "hipdnn_backend"},
        ),
        ([], set()),
    ],
)
def test_the_flag_reaches_the_parser_repeatably(tmp_path, argv, expected):
    args = parse_unified_args(
        [
            str(tmp_path),
            "--rocm-version",
            "7.14.0",
            # A repository source rather than --rocm-path, which would have the
            # parser shell out to clang to find its resource directory.
            "--rocm-libraries-dir",
            str(tmp_path),
            *argv,
        ]
    )
    assert _parse_skip_libraries(args.skip_libraries) == expected
