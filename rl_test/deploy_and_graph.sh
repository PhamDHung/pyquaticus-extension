#!/bin/bash
x=0
for dir in ~/responsible_v2/pyquaticus/rl_test/ray_test/*/ # list directories in the form "/tmp/dirname/"
do
dir=${dir%*/} # remove the trailing "/"

#echo "${dir}" # print everything after the final "/"
val=$((x*500))
python single_agent_deploy.py "${dir}/policies/agent-0-policy" "./actions/actions_$val.txt" "./scores/scores_$val.txt"
python heatmap.py "./actions/actions_$val.txt" "./graphs/images_$val"
x=$((x+1))
done
