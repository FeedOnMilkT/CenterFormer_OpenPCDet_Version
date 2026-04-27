#!/usr/bin/env bash

# Thin helper around torchrun for single-node DDP.
set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} --master_port=${PORT} train.py --launcher pytorch ${PY_ARGS}
