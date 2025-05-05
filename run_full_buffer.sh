#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")
HIDDENLAYERSIZES=(10 20 40 50)
LEARNINGRATES=(0.01 0.001 0.0001 0.00001)
DEFAULTMODEL="mlpregressor"
ACTIVATIONS=("relu" "tanh" "logistic" "identity")
CLUSTERTHRESHOLDS=(0.7 0.8 0.9)


run_models() {
	echo "TESTING THREE MODELS" >> out/all.log 2>&1
	echo "T parameter is: $1"

	for M in "${MODELS[@]}"; do
		echo "Running model: $M"
		echo "with t $1"

	done
}

run_learning_rates() {
	echo "TESTING LEARNING RATES" >> out/all.log 2>&1
	echo "T parameter is: $1"

	for LR in "${LEARNINGRATES[@]}"; do
		echo "Learning rate $LR"
		echo "With t $1"
	done
}

run_activations() {
	echo "TESTING ACTIVATION FUNCTIONS" >> out/all.log 2>&1
	echo "T parameter is: $1"

	for AC in "${ACTIVATIONS[@]}"; do
		echo "Function: $AC"
		echo "with t $1"

	done
}

run_cluster_thresholds() {
	echo "TESTING DIFFERENT THRESHOLDS" >> out/all.log 2>&1
	for T in "${CLUSTERTHRESHOLDS[@]}"; do
		echo "current running threshold: $T"
		run_models $T
		run_learning_rates $T
		run_activations $T

	done



}

case "$1" in
	allcluster)
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




