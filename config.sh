#!/bin/bash
set -e

PROJECT_ROOT=$(pwd)

case "$1" in
    --config)
        python "$PROJECT_ROOT/src/config/file_config.py" \
            --root "$PROJECT_ROOT"
        ;;

    --embed)
        python "$PROJECT_ROOT/src/processors/raw_process.py" \
            --root "$PROJECT_ROOT"
        ;;
   
esac