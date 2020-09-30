import pandas as pd
import string
import json
import jsonlines
import csv
import numpy as np

phemefile = pd.read_csv("PHEME_sampled_raw.csv")
pheme_subset = phemefile[["text", "topic", "posting_user_id"]]

tweet_id_list = []
for index, row in pheme_subset.iterrows():
    tweet_id_list.append(row['posting_user_id'])

print(len(tweet_id_list))
filename = "train_w_structure.json"
# print(tweet_id_list)

with open(filename, "r+") as f:
    with open('PHEME_realset.jsonl', 'w') as outfile:
        for line in f:
            data = json.loads(line)
            # print(f'sample set:{data["id_"]}')
            if data["id_"] in tweet_id_list:
                # print(f'extracted set:{data["id_"]}')
                tweet_id_list.remove(data["id_"])
                outfile.write(line)


if "525068915068923904" in tweet_id_list:
    print("yes")
