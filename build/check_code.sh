#!/bin/sh

python_files=$(git ls-files | grep .py)

echo ${python_files} | xargs pylint
