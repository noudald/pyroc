#!/bin/sh

python_files=$(git ls-files | grep .py | grep -v tests/)

echo ${python_files} | xargs pylint
