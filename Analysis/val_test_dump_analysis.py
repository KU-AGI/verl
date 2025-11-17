import json

from glob import glob
from pprint import pprint


task = "forward"

test_dump_path = "/data/verl/dumps/verl-dapo/refl_bonus_0.3/test/*.jsonl"
test_files = glob(test_dump_path)
test_files = sorted(test_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))[-10:]

val_dump_path = "/data/verl/dumps/verl-dapo/refl_bonus_0.3/val/*.jsonl"
val_files = glob(val_dump_path)
val_files = sorted(val_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))



for test_file in test_files:
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if task == "forward":
        test_data = test_data[0:1000]
    elif task == "retro":
        test_data = test_data[1000:2000]
    elif task == "reagent":
        test_data = test_data[2000:3000]
    for test_entry in test_data:
        gt = test_entry['gts'].split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        print()
        print("="*100)
        # print(test_entry["gts"])
        # print("-"*100)
        print(test_entry["output"])
        print("-"*100)
        print(gt)
        print("="*100)

        if not test_entry['acc']:
            print()



