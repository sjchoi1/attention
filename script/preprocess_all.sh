#!/bin/bash
WORKLOAD=(bc)
THREAD=(16)

cd /home/sjchoi/attention

for w in ${WORKLOAD[@]}
do
    for t in ${THREAD[@]}
    do 
        for i in {0..4}
        do
            python3 preprocess.py -workload ${w} -thread_cnt ${t} -idx ${i} &
        done
    done
done
