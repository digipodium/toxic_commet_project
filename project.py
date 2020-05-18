import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize  # tokenizer used when training TFIDF vectorizer
import pickle
import os
import pandas as pd


basepath = os.path.abspath(os.getcwd())
with open(basepath + '/models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
        tfidf_model = pickle.load(tfidf_file)
with open(basepath + '/models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
    logistic_toxic_model = pickle.load(logistic_toxic_file)
with open(basepath + '/models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
    logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)
with open(basepath + '/models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
    logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)
with open(basepath + '/models/logistic_insult.pkl', 'rb') as logistic_insult_file:
    logistic_insult_model = pickle.load(logistic_insult_file)
with open(basepath + '/models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
    logistic_obscene_model = pickle.load(logistic_obscene_file)
with open(basepath + '/models/logistic_threat.pkl', 'rb') as logistic_threat_file:
    logistic_threat_model = pickle.load(logistic_threat_file)


def add_comment(comment,predictions):
    if not os.path.exists('dataset.csv'):
        with open('dataset.csv','w') as file:
            file.write('comment,toxic,severe_toxic,hate,insult,obscene,threat\n')
    if os.path.exists('dataset.csv'):
        with open('dataset.csv','a') as file:
            file.write(f"""{comment},{predictions['pred_toxic']},{predictions['pred_severe_toxic']},{predictions['pred_identity_hate']},{predictions['pred_insult']},{predictions['pred_obscene']},{predictions['pred_threat']}\n""")
        return True
    else:
        return False

def analyse_message(msg):
    dict_preds = {}
    comment_term_doc = tfidf_model.transform([msg])
    dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]
    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)
    return dict_preds


st.title("Toxicity Prediction in text messages")
st.subheader("using machine learning")
st.write("Machine learning model loaded")

st.write("Please write/paste some message or comment below")
message = st.text_area("enter text")
checkraw = st.checkbox("view raw output")
button = st.button("submit")    
if  checkraw and button :
    results = analyse_message(message)
    st.write(results)
    add_comment(message,results)
else:
    results = analyse_message(message)
    add_comment(message,results)

st.write("database")

df = pd.read_csv("dataset.csv")
if st.checkbox("view raw database"):
    st.write(df)
 