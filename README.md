# DSP_2020_Final


## Crawling data for CNN
    python3 get_data.py -num 3


## Summarization demo
    python3 run.py -e 1


## Pretrained data

    wget https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
    wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt


## Testing for ROUGE
* First download the CNN_Dailymail from tensorflow dataset (Can be done in ROUGE_test.ipynb)
* Follow the steps in ROUGE_test.ipynb