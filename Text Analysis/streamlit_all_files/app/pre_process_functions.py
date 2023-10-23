import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

class PreProcess():

    def __init__(self, file_path):
        self.file_path = file_path
        self.text_df = pd.read_csv(file_path, encoding='latin')        
    
    def remove_punctuation(self, text):
        no_punct=[words for words in text if words not in string.punctuation]
        words_wo_punct=''.join(no_punct)
        return words_wo_punct
    
    def tokenize(self, text):
        split= word_tokenize(text)
        return split
    
    def remove_stopwords(self, text):
        stop_words = set(stopwords.words("english"))
        filtered_list = [word for word in text if word.casefold() not in stop_words]     
        return filtered_list            

    def lemmatize(self, tokens):
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lmtzr = WordNetLemmatizer()
        lem_list=[]
        for token, tag in pos_tag(tokens):
            lem_list.append(lmtzr.lemmatize(token, tag_map[tag[0]]))
        return lem_list


def generate_sentiment(sentiment_df):    
    sid = SentimentIntensityAnalyzer()    

    sentiment_df['like_scores'] = sentiment_df['liking_free_text'].apply(lambda x: sid.polarity_scores(str(x)))
    sentiment_df['meets_expectation_score'] = sentiment_df['meets_expectation_free_text'].apply(lambda x: sid.polarity_scores(str(x)))
    sentiment_df['prod_improve'] = sentiment_df['product_improvement_free_text'].apply(lambda x: sid.polarity_scores(str(x)))
    
    sentiment_df['like_compound']  = sentiment_df['like_scores'].apply(lambda score_dict: score_dict['compound'])
    sentiment_df['meets_expectation_compound']  = sentiment_df['meets_expectation_score'].apply(lambda score_dict: score_dict['compound'])
    sentiment_df['prod_improve_compound']  = sentiment_df['prod_improve'].apply(lambda score_dict: score_dict['compound'])    
    # sentiment_df['prod_improve_compound_class'] = sentiment_df['prod_improve_compound'].apply(lambda c: 'pos' if c >=0.5 else ('neu' if c>=-0.5 and c<=0.5 else 'neg'))

    return sentiment_df


def clean_data():
    # This should be argparse
    # pp_obj = PreProcess("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\Yogurt OSY vs chobani text.csv")
    # pp_obj = PreProcess("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\spade_ta\\spade_text.csv")
    pp_obj = PreProcess("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\bbq_tq_files\\bbq_text.csv")
    pp_df = pp_obj.text_df

    str_cols_ls = ["liking_free_text", "meets_expectation_free_text", "product_improvement_free_text"]
    # str_cols_ls = ["liking_free_text", "meets_expectation_free_text"]
    for col in str_cols_ls:
        pp_df[f"{str(col).lower()}_no_punc"]= pp_df[col].apply(lambda x: pp_obj.remove_punctuation(str(x)))
        pp_df[f"{str(col).lower()}_no_punc_tokenized"] = pp_df[f"{str(col).lower()}_no_punc"].apply(lambda x: pp_obj.tokenize(x.lower()))
        pp_df[f"{str(col).lower()}_no_punc_tokenized_no_stop"] = pp_df[f"{str(col).lower()}_no_punc_tokenized"].dropna().apply(lambda x: pp_obj.remove_stopwords(x))
        pp_df[f"{str(col).lower()}_no_punc_tokenized_no_stop_lemm"] = pp_df[f"{str(col).lower()}_no_punc_tokenized_no_stop"].dropna().apply(lambda x: pp_obj.lemmatize(x))

    pp_sentiment_df = generate_sentiment(pp_df)
    
    cols_to_keep = ['product_code','liking','liking_free_text','meets_expectation','meets_expectation_free_text','product_improvement_free_text','liking_free_text_no_punc_tokenized_no_stop_lemm', 'meets_expectation_free_text_no_punc_tokenized_no_stop_lemm', 'product_improvement_free_text_no_punc_tokenized_no_stop_lemm', 'like_compound', 'meets_expectation_compound', 'prod_improve_compound' ]
    # cols_to_keep = ['product_code','liking','liking_free_text','post_taste_pi','meets_expectation_free_text','liking_free_text_no_punc_tokenized_no_stop_lemm', 'meets_expectation_free_text_no_punc_tokenized_no_stop_lemm','like_compound', 'meets_expectation_compound']
    clean_df = pp_sentiment_df[cols_to_keep]
    
    clean_df.to_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\bbq_tq_files\\bbq_text_output.csv",index=False)    
    return clean_df

clean_data()
