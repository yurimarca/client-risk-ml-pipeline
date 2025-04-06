#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR=$(pwd)

# Get pyenv root and python version
PYENV_ROOT="$HOME/.pyenv"
PYTHON_VERSION=$(cat .python-version)

# Create the cron job command to run every 10 minutes
# We need to:
# 1. Set up pyenv environment
# 2. Activate the correct Python version
# 3. Run the script
CRON_CMD="*/3 * * * * export PYENV_ROOT=\"$PYENV_ROOT\"; export PATH=\"\$PYENV_ROOT/bin:\$PATH\"; eval \"\$(pyenv init --path)\"; eval \"\$(pyenv init -)\"; cd $PROJECT_DIR && pyenv shell $PYTHON_VERSION && python fullprocess.py >> $PROJECT_DIR/cron.log 2>&1"

# Add the cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo "Cron job has been set up to run using pyenv environment: $PYTHON_VERSION"
echo "Logs will be written to $PROJECT_DIR/cron.log" 