#!/usr/bin/env bash
set -e

MODELS=("linearregression" "supportvectorregression" "mlpregressor")
HIDDENLAYERSIZES=(10 20 40 50)
LEARNINGRATES=(0.01 0.001 0.0001 0.00001)
DEFAULTMODEL="mlpregressor"

run_models() {
	echo "Called run_models"
}

run_learning_rates() {
	echo "Testing Learning Rates"
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
	echo "Running: $0 [models|lrs|all]"
	exit 1
	;;
esac


# open the .log files to see (comment this if not on windows)
for file in results_*.log; do
	notepad "$file" &
done


