#!/usr/bin/env bash
set -euo pipefail

python -m mlops.train --config configs/train_config.yaml
