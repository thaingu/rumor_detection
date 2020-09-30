from tabulate import tabulate
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from urllib.parse import urlparse, unquote
import torch
import json
import fire
import sys
import os

checkpoint = "detector-base.pt"
device = 'cpu'

# Set up detector
checkpoint = "detector-base.pt"
device = 'cpu'
if checkpoint.startswith('gs://'):
    print(f'Downloading {checkpoint}', file=sys.stderr)
    subprocess.check_output(['gsutil', 'cp', checkpoint, '.'])
    checkpoint = os.path.basename(checkpoint)
    assert os.path.isfile(checkpoint)

print(f'Loading checkpoint from {checkpoint}')
data = torch.load(checkpoint, map_location='cpu')
model_name = 'roberta-large' if data['args']['large'] else 'roberta-base'
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

model.load_state_dict(data['model_state_dict'])
model.eval()


def eval_query(query):
    tokens = tokenizer.encode(query)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.max_len - 2]
    used_tokens = len(tokens)
    tokens = torch.tensor([tokenizer.bos_token_id] +
                          tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()

    result = json.dumps(dict(
        all_tokens=all_tokens,
        used_tokens=used_tokens,
        real_probability=real,
        fake_probability=fake
    )).encode()
    return real, fake


temp_corp = {}
temp_sent = {}

# 0 is fake machine, 1 is real human
y_true = []
y_pred = []
temp_sent['predicted_label'] = []
temp_sent['human_probability'] = []
temp_sent['machine_probability'] = []
temp_sent['text'] = []
line_query = []
for cnt, line in enumerate(rumour_list):
    if '|' in line or '>' in line or '<' in line or len(line) < 30 or len(line) > 550:
        continue
    else:
        y_true.append(0)
        real, fake = eval_query(line)
        temp_sent['human_probability'].append(real)
        temp_sent['machine_probability'].append(fake)
        temp_sent['text'].append(line)
        real_int = round(real)
        y_pred.append(real_int)
        if real_int:
            temp_sent['predicted_label'].append("human")
            line_query.append(
                f"Line:{cnt} Human Real_Probability:{real} Fake_Probability:{fake} Text:{line}")
        else:
            temp_sent['predicted_label'].append("machine")
            line_query.append(
                f"Line:{cnt} Machine Real_Probability:{real} Fake_Probability:{fake} Text:{line}")
scores = precision_recall_fscore_support(y_true, y_pred, average='micro')
temp_corp['machine_classifier_precision'] = scores[0]
temp_corp['machine_classifier_recall'] = scores[1]
temp_corp['machine_classifier_fscore'] = scores[2]
accuracy = accuracy_score(y_true, y_pred)
temp_corp['machine_classifier_accuracy'] = accuracy

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
corpus_df = pd.DataFrame(temp_corp, index=[0])
sentence_df = pd.DataFrame(temp_sent)

phemefile = pd.read_csv("PHEME_sampled_raw.csv")
pheme_subset = phemefile[["text", "topic", "posting_user_id"]]
