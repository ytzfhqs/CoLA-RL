from collections import Counter

import pandas as pd
from sklearn.metrics import matthews_corrcoef
from utils import read_jsonl_offline, save_json

prompt = """
Decide whether the following sentence is grammatically acceptable or not. If it is grammatically correct, answer "acceptable". If not, answer "unacceptable". Only output "acceptable" or "unacceptable", and do not output any other information.

Sentence: {sentence}

Your answer:
"""

think_prompt = "<think>\n{reasoning}\n</think>\n\n{answer}"

cola_res = read_jsonl_offline("cola_data/R1_infer_train.jsonl")
train_data = pd.read_csv(
    "cola_data/in_domain_train.tsv",
    sep="\t",
    header=None,
    names=["source", "label", "first_label", "text"],
)

res_map = {}
for cr in cola_res:
    custom_id = cr['custom_id']
    content = cr['response']['body']['choices'][0]['message']['content'].strip()
    reasoning = cr['response']['body']['choices'][0]['message']['reasoning_content'].strip()
    res_map[custom_id] = {'content': content, 'reasoning': reasoning}

sft_data = []
y_true = []
y_pred = []
label_count = []
for idx, row in enumerate(train_data.itertuples(index=False)):
    custom_id = f"request-{idx}"
    label = 'acceptable' if row.label == 1 else 'unacceptable'
    y_true.append(label)
    y_pred.append(res_map[custom_id]['content'])
    if label == res_map[custom_id]['content']:
        label_count.append(label)
        reasoning = res_map[custom_id]['reasoning']
        sft_data.append({'instruction': prompt.format(sentence=row.text), 'output': think_prompt.format(answer = label, reasoning = reasoning)})
print(len(res_map))
print(Counter(label_count))
print(matthews_corrcoef(y_true, y_pred))
save_json("data/cola_sft/cola_train_cot.json", sft_data)