#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status

chsh -s $(which zsh) root || true
if ! command -v starship &> /dev/null; then
  echo "Installing Starship prompt..."
  curl -sS https://starship.rs/install.sh | sh -s -- -y
fi
python3 -m pip install pkgconfig
HDF5_MPI=ON HDF5_PKGCONFIG_NAME="hdf5" python3 -m pip install h5py --no-build-isolation --no-binary=h5py
python3 -m pip install scifem --no-build-isolation --no-binary=scifem
python3 -m pip install -e .[all]
pre-commit install

# 1. Install Claude Code
echo "Installing Claude..."
curl -fsSL https://claude.ai/install.sh | bash

# 2. Install Node.js v22
echo "Installing Node.js v22..."
curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO bash -
$SUDO apt-get install -y nodejs

# 3. Install mattpocock skills
echo "Installing mattpocock/skills..."
npx skills@latest add mattpocock/skills

# 4. Install Superpowers Plugin
echo "Installing Superpowers plugin..."
claude plugin install superpowers@claude-plugins-official

# 5. Install Writing Agents
echo "Installing Academic Writing Agents..."
claude plugin marketplace add andrehuang/academic-writing-agents
# claude plugin install andrehuang@academic-writing-agents

echo "Installing Humanizer plugin..."
claude plugin marketplace add blader/humanizer
# claude plugin install https://github.com/blader/humanizer

# 3. Print success message
echo "Container setup complete!"
