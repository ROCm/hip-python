#!/usr/bin/env python3

template="""\
# --extra-index-url https://test.pypi.org/simple/
hip-python=={ROCM_VER}.*
hip-python-as-cuda=={ROCM_VER}.*
rocm-llvm-python=={ROCM_VER}.*
"""

pyproject_toml_optional_deps = ""

for v in (
    "6.1.2",
    '6.1.0',
    '6.0.0',
):
    with open(f"requirements/requirements-{v}.txt", "w") as outfile:
        outfile.write(template.format(ROCM_VER=v))

    # note: insert into 'pyproject.toml' file's '[tool.setuptools.dynamic]' section
    key = v.replace(".","_")
    pyproject_toml_optional_deps += f"optional-dependencies.rocm_{key} = {{ file = [\"requirements/requirements-{v}.txt\"] }}\n"

print(pyproject_toml_optional_deps)