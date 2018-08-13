#!/bin/bash

PWD=`pwd`


port=5001
nRuns=100
hostname="localhost"

# Allowed values for algorithm parameter(case-sensitive)
# 1. epsilon-greedy 
# 2. UCB 
# 3. KL-UCB 
# 4. Thompson-Sampling
# 5. rr


algorithm="rr"
horizon=100000
epsilon=0.1
numArms=5
randomSeed=3

banditFile="$PWD/data/instance-bernoulli-5.txt"


SERVERDIR=./server
CLIENTDIR=./client

OUTPUTFILE=$PWD/serverlog.txt

pushd $SERVERDIR
cmd="./startserver.sh 5 100000 5001 ./data/instance-bernoulli-5.txt 3 serverlog.txt &"
# echo $cmd
$cmd 
popd

sleep 1

pushd $CLIENTDIR
cmd="./startclient.sh 5 100000 localhost 5001 3 rr 0.1 &"
#echo $cmd
$cmd > /dev/null 
popd

