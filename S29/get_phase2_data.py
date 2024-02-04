import json
import pandas as pd
import pickle
import os
from datasets import Dataset, load_dataset

def prep_data(df):
    df_assistant = df[(df.role == "assistant") & (df["rank"] == 0.0)].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["Answer"] = df_assistant["text"].values

    inputs = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)

    df_assistant["Question"] = inputs

    df_assistant = df_assistant[df_assistant.lang == "en"]

    df_assistant = df_assistant[
        ["Question", "Answer", "message_id"]
    ].rename(columns={"message_id": "Ids"})

    return df_assistant


def main():
    dataset_name = "OpenAssistant/oasst1"
    train_ds, val_ds = load_dataset(dataset_name, split=['train', 'validation'])
    df_train = prep_data(train_ds.to_pandas())
    df_val = prep_data(val_ds.to_pandas())

    train_list = list()
    val_list = list()
    max_tokens = 30
    for _, row in df_train.iterrows():
        if len(row['Question'])>max_tokens:
            continue
        if len(row['Answer'])>max_tokens:
            continue
        train_list.append({"ImageUrl": 0, "Question": row['Question'], "Answer": row["Answer"]})
    for _, row in df_val.iterrows():
        if len(row['Question'])>max_tokens:
            continue
        if len(row['Answer'])>max_tokens:    
            continue
        val_list.append({"ImageUrl": 0, "Question": row['Question'], "Answer": row["Answer"]})

    with open(r"./data/llava_instruct_150k.json","r")as j:
        i150_json = json.load(j)
   

    i150k = list()
    for idx, data in enumerate(i150_json):
        image = data['image']
        image_url = 'http://images.cocodataset.org/train2017/' + image
        id_ = data["id"]
        iterator = iter(data['conversations'])
        for i in iterator:
            ques = i
            ans = next(iterator)
            if (len(ques["value"])>max_tokens or len(ans["value"])>max_tokens):
                continue
            if ques["from"] == "human" and ans["from"] == "gpt":
                i150k.append({"ImageUrl": image_url, 
                            "Question": ques["value"].replace("<image>\n","").replace("<image>",""), 
                            "Answer": ans["value"]})
    train_i150k, val_i150k = i150k[:int(0.95*len(i150k))], i150k[int(0.95*len(i150k)):]
    train_list.extend(train_i150k)
    val_list.extend(val_i150k)

    with open('./data/train.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open('./data/val.pkl', 'wb') as f:
        pickle.dump(val_list, f)

if __name__ == "__main__":
    main()