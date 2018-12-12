#!/bin/bash

# the situations when we need to move config file from run_specific file first including:
# rerun a previous config file
if [ $# -eq 0 ]
  then
    echo "Use cur dir's config file"
    ./RNN.py
  else
    echo "with parameters {$1}"
    ./RNN.py "$1"
fi

