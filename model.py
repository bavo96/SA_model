import os
import sys
import gdown

import numpy as np
import datasets
import py_vncorenlp

from datasets import Dataset
from tqdm.notebook import tqdm
from nltk.corpus.reader import ConllCorpusReader

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from datasets import Features, Sequence, ClassLabel, Value, DatasetDict
from datasets import load_metric

def download_model(url, output='./saved_model/'):
    gdown.download_folder(url=url,quiet=True, use_cookies=False, remaining_ok=True)

if not os.path.isdir("./saved_model_2/"):
    download_model("https://drive.google.com/drive/folders/1HnGV6K7ZIj_V_qT5J6XaF-6svL6jxS7m")

class SentimentAnalysis:
    def __init__(self, model_url):
        self.model_url = model_url
        if not os.path.exists("./vncorenlp"):
            os.mkdir("./vncorenlp")
            print('Downloading vncorenlp...')
            py_vncorenlp.download_model(save_dir='./vncorenlp')

        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./vncorenlp')
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        model = AutoModelForSequenceClassification.from_pretrained(self.model_url, num_labels=2)
        model.cuda().eval()

    def inference(self, text):
        text = text.lower()
        text = " ".join(rdrsegmenter.word_segment(text))
        input = tokenizer(text, return_token_type_ids=False, truncation=True, return_tensors='pt')
        input_ids = input['input_ids'].cuda()
        attention_mask = input['attention_mask'].cuda()
        logits = model(input_ids).logits.cpu().detach().numpy()
        pred = np.argmax(logits, axis=1)[0]
        if pred == 0:
            return "negative"
        else:
            return "positive"
