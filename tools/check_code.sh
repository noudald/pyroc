#!/bin/bash

PYTHONFILES=($(git ls-files | grep ".*\.py$" | grep -v tests/ | tr "\n" " "))

echo "Check code script."
echo -e "Checking files:\n${PYTHONFILES[@]}\n"

FAILURE=0

echo "# Pylint tests"
pylint ${PYTHONFILES[@]}

if [ $? -eq 0 ]; then
    echo -e "Passed\n"
else
    echo -e "Failed\n"
    FAILURE=$((FAILURE+1))
fi

echo "# Mypy tests"
mypy --config-file=tools/mypy.ini ${PYTHONFILES[@]}

if [ $? -eq 0 ]; then
    echo -e "\nPassed"
else
    echo -e "\nFailed"
    FAILURE=$((FAILURE+1))
fi

exit ${FAILURE}
