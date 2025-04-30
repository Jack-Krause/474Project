#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")

for M in "${MODELS[@]}"; do
	echo "Running model: $M"

	python main.py \
		-model_name "$M" \
		> "results_${M}.log" 2>&1

done


