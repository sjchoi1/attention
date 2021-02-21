#!/bin/bash
WORKLOAD=(svm)
THREAD=(1)
DATA_DIR=/home/sjchoi/attention/data
CUR_DIR=/home/sjchoi/attention/script

for w in ${WORKLOAD[@]}
do 
    for t in ${THREAD[@]}
    do
	cd ${CUR_DIR}
        for i in {0..9}
        do
            in_file_name=${w}_raw_t${t}_${i}.txt
            out_file_name=${i}.csv
            in_file_path=${DATA_DIR}/${w}/raw/t${t}/${in_file_name}
            out_csv_path=${DATA_DIR}/${w}/csv/t${t}/${out_file_name}
            python3 raw_to_csv.py ${in_file_path} ${out_csv_path}
        done
	cd ${DATA_DIR}/${w}/csv/t${t}
	cat 0.csv 1.csv 2.csv 3.csv 4.csv 5.csv 6.csv > train.csv
	cat 7.csv 8.csv 9.csv > val.csv
    done
done
