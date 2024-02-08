#!/bin/bash
sudo apt update

# Install poetry
pipx install poetry --pip-args '--no-cache-dir --force-reinstall'

# Install project dependencies
poetry install
