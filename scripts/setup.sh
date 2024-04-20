#!/bin/bash

cd $(pwd)

python -m rjhmc-env ./envs/
source ./envs/rjhmc-env/bin/activate
pip install -r ./envs/requirements-rjhmc.txt
cp -f ./envs/numpyro-changes/* ./envs/rjhmc-env/lib/python3.8/site-packages/numpyro/infer/
deactivate

exit 0
