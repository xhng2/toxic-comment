import numpy as np
import re
import string
from sklearn.metrics import roc_auc_score

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')

# Preprocessing
eng_stopwords = stopwords.words('english')
def tokenize(text, stopwords=None):
    regex = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    tokens = regex.sub(r' \1 ', text).split()
    if stopwords:
        tokens = [word for word in tokens if word.lower() not in stopwords]
    return tokens

def tokenize_with_stopw(text, stopwords=eng_stopwords):
    return tokenize(text, stopwords)
    
def tokenize_and_lemma(text, stopwords=None):
    lemma = WordNetLemmatizer()
    tokens = tokenize(text, stopwords)
    tokens = [lemma.lemmatize(word.lower()) for word in tokens]
    return tokens

def preprocess_text(text):
    # 1. Convert to lowercase
    clean_text = text.lower()
    # 2. Remove urls
    clean_text = re.sub(r'https?.*(?=[\s])', '', clean_text,
                  flags=re.IGNORECASE)
    # 3. Remove IP addresses
    clean_text = re.sub(r'\d+\.\d+\.\d+\.\d+', '', clean_text,
                  flags=re.IGNORECASE)
    # 4. Remove user names
    clean_text = re.sub(r'\[\[User\:.*\|', '', clean_text,
                  flags=re.IGNORECASE)
    return clean_text


# Evaluation
def eval_pred(actuals, preds, class_labels):
    '''Compute mean column-wise ROC AUC'''
    # Filter for those in test set with labels
    mask = (actuals[class_labels].min(axis=1)!=-1).values
    filt_actuals = np.array([arr for arr, m in zip(actuals[class_labels].values, mask) if m])
    filt_preds = np.array([arr for arr, m in zip(preds, mask) if m])
    
    scores = []
    for idx, label in enumerate(class_labels):
        score = roc_auc_score(filt_actuals[:, idx], filt_preds[:, idx])
        scores.append(score)
#         print(label, score)
    mean_score = np.mean(scores)
    print(f'Mean ROC-AUC: {mean_score}')
    return mean_score