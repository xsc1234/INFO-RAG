### Install
```
pip install -r requirements.text
```

### Downloar wikipedia passage data from DPR
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

### Make unsupervised data
```
python make_unsupervised_data/make_data.py --psgs_data_path your_wikipedia_path --uns_data_path your_data_save_path
```

### Fine-tune LLMs via LoRA follow INFO-RAG
```
bash train_info_rag.sh
```
