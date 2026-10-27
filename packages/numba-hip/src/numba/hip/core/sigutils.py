# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import ast

from numba.core import types, typing


def is_signature(sig):
    """
    Return whether *sig* is a potentially valid signature
    specification (for user-facing APIs).
    """
    return isinstance(sig, (str, tuple, typing.Signature))


def _parse_signature_string(signature_str):
    """Parse a signature expression using the public Numba type namespace."""
    try:
        expression = ast.parse(signature_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid signature expression: {signature_str!r}") from exc

    def evaluate(node):
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id not in types.__dict__:
                raise ValueError(f"unknown type name: {node.id!r}")
            return types.__dict__[node.id]

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("signature calls must name a Numba type")
            func = evaluate(node.func)
            args = [evaluate(arg) for arg in node.args]
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ValueError("signature calls do not support **kwargs")
                kwargs[keyword.arg] = evaluate(keyword.value)
            return func(*args, **kwargs)

        if isinstance(node, ast.Tuple):
            return tuple(evaluate(element) for element in node.elts)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, str, bool)) or node.value is None:
                return node.value
            raise ValueError(f"unsupported signature constant: {node.value!r}")

        if isinstance(node, ast.Subscript):
            return evaluate(node.value)[evaluate_slice(node.slice)]

        raise ValueError(
            f"unsupported signature expression: {ast.unparse(node)!r}"
        )

    def evaluate_slice(node):
        if isinstance(node, ast.Slice):
            return slice(
                evaluate(node.lower) if node.lower is not None else None,
                evaluate(node.upper) if node.upper is not None else None,
                evaluate(node.step) if node.step is not None else None,
            )
        if isinstance(node, ast.Tuple):
            return tuple(evaluate_slice(element) for element in node.elts)
        return evaluate(node)

    return evaluate(expression.body)


def normalize_signature(sig):
    """
    From *sig* (a signature specification), return a ``(args, return_type)``
    tuple, where ``args`` itself is a tuple of types, and ``return_type``
    can be None if not specified.
    """
    if isinstance(sig, str):
        parsed = _parse_signature_string(sig)
    else:
        parsed = sig
    if isinstance(parsed, tuple):
        args, return_type = parsed, None
    elif isinstance(parsed, typing.Signature):
        args, return_type = parsed.args, parsed.return_type
    else:
        raise TypeError(
            "invalid signature: %r (type: %r) evaluates to %r "
            "instead of tuple or Signature"
            % (sig, sig.__class__.__name__, parsed.__class__.__name__)
        )

    def check_type(ty):
        if not isinstance(ty, types.Type):
            raise TypeError(
                "invalid type in signature: expected a type "
                "instance, got %r" % (ty,)
            )

    if return_type is not None:
        check_type(return_type)
    for ty in args:
        check_type(ty)

    return args, return_type
