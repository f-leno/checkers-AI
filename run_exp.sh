#
# Shell script for running Tic Tac Toe experiments, the arguments are:
# $1 = initial trial
# $2 = final trial
# $3 = name of source/algorithm to be executed
# $4 = type of evaluation (see arg train_env in main.py)
# $5 = Should the agent count steps against the simulated agent?
# $6 = parameter y
# $7 = parameter x
# 
# Author: Felipe Leno <f.leno@usp.br>
# Example: sh run_exp.sh 1 2 qLearning:QLearningAgent regular 1 200 400

for i in `seq $1 $2`
do
	echo "RUN " $i
    python3 ./main.py --trial $i --algorithm $3 --train_env $4 --recording_mode reward --count_fict $5 --number_expert $6 --number_simulated $7 
done
