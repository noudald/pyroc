#!/bin/bash

PYTHONFILES=($(git ls-files | grep .py | grep -v tests/ | tr "\n" " "))

echo "Check code script."
echo -e "Checking files:\n${PYTHONFILES[@]}\n"

FAILURE=0

echo "# Pylint tests"
pylint ${PYTHONFILES[@]}

if [ $? -eq 0 ]; then
    echo -e "Passed\n"
else
    echo -e "Failed\n"
    FAILURE=1
fi

echo "# Mypy tests"
mypy --ignore-missing-imports ${PYTHONFILES[@]}

if [ $? -eq 0 ]; then
    echo "Passed"
else
    echo "Failed"
    FAILURE=1
fi

exit ${FAILURE}
