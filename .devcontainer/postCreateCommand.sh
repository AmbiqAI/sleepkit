#!/bin/bash
sudo apt update

# Install poetry
pipx install poetry==1.6.1 --pip-args '--no-cache-dir --force-reinstall'

# Install project dependencies
poetry install
