#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")
HIDDENLAYERSIZES=(10 20 40 50)
LEARNINGRATES=(0.01 0.001 0.0001 0.00001)
DEFAULTMODEL="mlpregressor"
ACTIVATIONS=("relu" "tanh" "logistic" "identity")

run_models() {
	echo "TESTING THREE MODELS" >> out/all.log 2>&1
	for M in "${MODELS[@]}"; do
		echo "Running model: $M"

		python main.py \
			-model_name "$M" \
			>> "out/all.log" 2>&1
	done
}

run_learning_rates() {
	echo "TESTING LEARNING RATES" >> out/all.log 2>&1
	for LR in "${LEARNINGRATES[@]}"; do
		python main.py \
			-model_name "$DEFAULTMODEL" \
			-learning_rate "$LR" \
			>> "out/all.log" 2>&1
	done
}

run_activations() {
	echo "TESTING ACTIVATION FUNCTIONS" >> out/all.log 2>&1
	for AC in "${ACTIVATIONS[@]}"; do
		echo "Function: $AC"

		python main.py \
			-model_name "mlpregressor" \
			-activation_function "$AC" \
			>> "out/all.log" 2>&1
	done
}

case "$1" in
	activations | act)
		run_activations
		;;
	models)
		run_models
		;;
	lrs|learningrates)
		run_learning_rates
		;;
	all|"")
		run_models
		run_learning_rates
		run_activations
		;;
	*)
		echo "Possible arguments are: $0 [models|lrs|all] (invalid cmd argument)"
		exit 1
	;;
esac




