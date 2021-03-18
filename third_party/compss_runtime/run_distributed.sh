#!/bin/bash

# Clean worker node
ssh compss@compsscito1 "ps aux | grep \"python\|java\|bash.*[c]ompss\" | awk '{print \$2}' | xargs kill -9"

# Clean master node
ps aux | grep "python\|java\|bash.*[c]ompss" | awk '{print $2}' | xargs kill -9

# Copy project to worker node
scp -r ../../../pyeddl compss@compsscito1:
ssh compss@compsscito1 "cd pyeddl && python3 setup.py install"

# Rebuild pyeddl
cd ../../
python3 setup.py install
cd ./third_party/compss_runtime/

# Run
runcompss -d --lang=python --jvm_workers_opts="-Dcompss.worker.removeWD=false" --python_interpreter=python3 --project=./xml/project.xml --resources=./xml/resources.xml eddl_train_batch_compss.py --num_workers 4 --num_epochs 1 --num_workers_batch_size 250
