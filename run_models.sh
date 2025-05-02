#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")
HIDDENLAYERSIZES=(10 20 40 50)
LEARNINGRATES=(0.01 0.001 0.0001 0.00001)
DEFAULTMODEL="mlpregressor"

run_models() {
	echo "Called run_models"
	for M in "${MODELS[@]}"; do
		echo "Running model: $M"

		python main.py \
			-model_name "$M" \
			> "out/results_${M}.log" 2>&1
	done
}

run_learning_rates() {
	echo "Testing Learning Rates"
	for LR in "${LEARNINGRATES[@]}"; do
		python main.py \
			-model_name "$DEFAULTMODEL" \
			-learning_rate "$LR" \
			> "lr_res_${LR}.log" 2>&1
	done
}

case "$1" in
	models)
		run_models
		;;
	lrs|learningrates)
		run_learning_rates
		;;
	all|"")
		run_models
		run_learning_rates
		;;
	*)
		echo "Possible arguments are: $0 [models|lrs|all] (invalid cmd argument)"
		exit 1
	;;
esac


# open the .log files to see (comment this if not on windows)
for file in out/results_*.log; do
	notepad "$file" &
done
exit 1


