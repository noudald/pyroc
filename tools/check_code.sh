#!/bin/sh

PYTHONFILES=$(git ls-files | grep .py | grep -v tests/)

echo "Check code script."
echo "Checking files:\n${PYTHONFILES}\n"

echo "# Pylint tests"
echo ${PYTHONFILES} | xargs pylint

if [ $? -eq 0 ]; then
    echo "Passed\n"
fi

echo "# Mypy tests"
echo ${PYTHONFILES} | xargs mypy --ignore-missing-imports

if [ $? -eq 0 ]; then
    echo "Passed"
fi
