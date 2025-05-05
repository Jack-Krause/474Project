#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")
HIDDENLAYERSIZES=(10 20 40 50)
LEARNINGRATES=(0.01 0.001 0.0001 0.00001)
DEFAULTMODEL="mlpregressor"
ACTIVATIONS=("relu" "tanh" "logistic" "identity")
CLUSTERTHRESHOLDS=(0.3 0.2 0.1)


run_models() {
	echo "TESTING THREE MODELS" >> "out/MODELs.log" 2>&1
	echo "Models T parameter is: $1"

	for M in "${MODELS[@]}"; do
		echo "Running model: $M, T: $1"

		python test.py \
			>> "out/MODELs.log" 2>&1
	done
}

run_learning_rates() {
	echo "TESTING LEARNING RATES" >> "out/LRs.log" 2>&1
	echo "LR T parameter is: $1"

	for LR in "${LEARNINGRATES[@]}"; do
		echo "Running default with LR: $LR, T: $1"
		python test.py \
			>> "out/LRs.log" 2>&1
	done
}

run_activations() {
	echo "TESTING ACTIVATION FUNCTIONS" >> "out/ACTs.log" 2>&1
	echo "Activations T parameter is: $1"

	for AC in "${ACTIVATIONS[@]}"; do
		echo "Function: $AC, T: $1"

		python test.py \
			>> "out/ACTs.log" 2>&1
	done
}

run_cluster_thresholds() {
	echo "TESTING DIFFERENT THRESHOLDS"
	for T in "${CLUSTERTHRESHOLDS[@]}"; do
		echo "current running threshold: $T"
		run_models $T
		run_learning_rates $T
		run_activations $T

	done



}

case "$1" in
	allcluster|thresholds)
		run_cluster_thresholds
		;;
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




