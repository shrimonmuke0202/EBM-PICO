from flair.data import Sentence
from flair.models import SequenceTagger

from tqdm import tqdm
import json

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from seqeval.metrics import classification_report

columns = {0: 'text', 1: 'ner'}
corpus: Corpus = ColumnCorpus('data/', columns,
                              train_file='train.txt',
                              dev_file='dev.txt',
                              test_file='test.txt'
                              )

# load the model you trained
model_mean = SequenceTagger.load('/model/best-model.pt')

result_mean = model_mean.evaluate(corpus.test, gold_label_type='ner',mini_batch_size=4, out_path=f"/model/test.tsv")
# print(result_mean)

test_r_lsts = open("/model/test.tsv").readlines()

y_true = []
y_pred = []
token_level_true = []
token_level_pred = []



for ind,test_l in enumerate(test_r_lsts):
    if test_l.split(" ")[0] == '\n':
        y_true.append(token_level_true)
        y_pred.append(token_level_pred)
        token_level_true = []
        token_level_pred = []
    else:
        split_r = test_l.split(" ")
        true_ = test_l.split(" ")[1]
        pred_ = test_l.split(" ")[-1].rstrip("\n")
        if true_.startswith("I"):
            if true_.split("-")[-1] == "EFFECTS":
                true_i = "B-" + true_.split("-")[1] + '-' + true_.split("-")[-1]
                token_level_true.append(true_i)
            else:
                true_i = "B-" + true_.split("-")[-1]
                token_level_true.append(true_i)
        if pred_.startswith("I"):
            if pred_.split("-")[-1] == "EFFECTS":
                pred_i = "B-" + pred_.split("-")[1] + '-' + pred_.split("-")[-1]
                token_level_pred.append(pred_i)
            else:
                pred_i = "B-" + pred_.split("-")[-1]
                token_level_pred.append(pred_i) 
        if pred_ == "O":
            token_level_pred.append(pred_)
        if true_ == "O":
            token_level_true.append(true_)
    
        if pred_.startswith("B"):
            token_level_pred.append(pred_)
        if true_.startswith("B"):
            token_level_true.append(true_)
            

class_R = classification_report(y_true, y_pred,digits=4,mode='strict')
print(class_R)