### Install Packages

```python
  pip3 install flair
  pip3 install seqeval
```
 
### Datasets
Please Download Datset from [Link](https://github.com/bepnye/EBM-NLP/) and followed the method describe in that link to get the CONLL format.
and save the data in the Data directory
### Run the following code to train the model
```python
    !python3 train.py --dataset_path Data/ \
--data_train train.txt\
--data_test test.txt\
--data_dev dev.txt\
--output_dir model \
--model_name_or_path michiyasunaga/BioLinkBERT-large \
--layers -1\
--subtoken_pooling first_last\
--hidden_size 256\
--learning_rate 5e-5\
--num_epochs 10 \
--use_crf False
```
### Inference
Run the following scrpit to test the best model
```python
  python3 evaluate.py
```
