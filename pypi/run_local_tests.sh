#!/bin/bash
# Run tests against local Julia module
export DUALPERSPECTIVE_USE_LOCAL=true

# Run from pypi directory
cd "$(dirname "$0")"

# Run all existing tests with the local version
python -m pytest tests/ -v
