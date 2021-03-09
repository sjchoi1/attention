import csv
import sys
import os

if len(sys.argv) != 3:
    print("usage: python3 raw_to_csv [in_file_path] [out_csv_path]")
    sys.exit()

raw = []
tid_to_idx = {}
cur_tid = 0

with open(sys.argv[1], 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    for line in stripped:
        if "$$$" in line:
            addr_ = line.split(" ")[-1]
            pc_ = line.split(" ")[-2]
            tid_ = line.split(" ")[-3]

            if tid_ not in tid_to_idx:
                tid_to_idx[tid_] = cur_tid
                cur_tid += 1

            raw.append([tid_to_idx[tid_], pc_, addr_])

os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
f = open(sys.argv[2], 'w', newline='')
wr = csv.writer(f)
for r in raw:
    wr.writerow(r)
f.close()
