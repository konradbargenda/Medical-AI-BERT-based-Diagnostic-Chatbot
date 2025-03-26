# Medical-AI-BERT-based-Diagnostic-Chatbot
#Medical AI

This project is a python programmed BERT-based model that can be used as a preliminary diagnostic tool. We ran the code and completed the project on Google Colab and uploaded the final notebook on the Github repository.

## The following datasets were used and must be imported to successfully run the code
Dataset 1: medical_conversations
source: https://www.kaggle.com/datasets/artemminiailo/medicalconversations2disease

Dataset 2: Disease_symptom_and_patient_profile_dataset (combined)
source:https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset


##Installation
Use the package manager [pip] to install the necessary libraries

```bash
!pip install scikit-learn
!pip install torch
!pip install datasets
!pip install transformers
!pip install gradio
nlrk.download(‘punkt’)
nlrk.download(stopwords)
nlrk.download(wordnet)
nlrk.download(‘punkt_tab’)

## Usage

```python
import pandas as pd
Import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import  stopwords
from nltk.stem import  WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import gradio as gr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
```
Link to original BERT model program: 
https://github.com/huggingface/notebooks
https://huggingface.co/docs/transformers/training
https://huggingface.co/docs/transformers/index
