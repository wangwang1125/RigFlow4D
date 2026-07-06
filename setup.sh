#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-rigflow4d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_DEV="${INSTALL_DEV:-1}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists."
    else
        echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    conda activate "$ENV_NAME"
else
    echo "Conda was not found; installing into the current Python environment."
fi

python -m pip install --upgrade pip

if [ "$INSTALL_DEV" = "1" ]; then
    python -m pip install -r requirements-dev.txt
else
    python -m pip install -r requirements.txt
fi

echo "RigFlow4D environment is ready."

