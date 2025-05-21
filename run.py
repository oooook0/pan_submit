from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np
import torch

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utility import ContextualRelevanceFeatures, NaturalTextPatternAnalyzer
from biscope_utility import detect_single_sample, MODEL_ZOO

from lexicalrichness import LexicalRichness

import textstat
import spacy

import swifter
import re

import os

import argparse
import sys

import joblib

#from huggingface_hub import login

#login(token='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')

nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

SELECTED_FEATURES =  ['biscope_feature_28', 'avg_content_density', 'biscope_feature_20',
       'biscope_feature_36', 'biscope_feature_44', 'sent_len_consistency',
       'biscope_feature_12', 'biscope_feature_52', 'biscope_feature_4',
       'length_cv', 'biscope_feature_60', 'biscope_feature_68',
       'biscope_feature_0', 'hapax_rate', 'biscope_feature_8',
       'biscope_feature_16', 'biscope_feature_7', 'CD', 'biscope_feature_24',
       'VB', 'biscope_feature_15', 'biscope_feature_32', 'biscope_feature_23',
       'PRP', 'entity_diversity', 'biscope_feature_31', 'tfidf_variance',
       'biscope_feature_40', 'pronoun_to_referent_ratio', 'biscope_feature_39',
       'biscope_feature_48', 'complex_verb_cnt',
       'unique_words_per_sentence_stdev', 'biscope_feature_47', 'flesch_score',
       'RB', 'biscope_feature_3', 'unique_pronouns', 'biscope_feature_56',
       'marker_diversity', 'biscope_feature_55', 'adverb_cnt',
       'unique_words_relative', 'CC', 'biscope_feature_11',
       'biscope_feature_64', 'avg_word_cnt', 'WRB', 'MD',
       'biscope_feature_63']

def count_adverbs(doc):
    """Counts the number of adverbs in a given text."""

    adverb_count = 0
    for word in doc:
        if word.tag_.startswith('RB'):  # Adverbs are tagged with 'RB'
            adverb_count += 1
    return adverb_count

def hapax_legomenon_rate(text):
    """Calculates the Hapax Legomenon Rate of a text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The Hapax Legomenon Rate.
    """

    words = re.findall(r'\w+', text.lower())
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    hapax_count = sum(1 for count in word_counts.values() if count == 1)
    total_words = len(words)

    return hapax_count / total_words

def get_sentence_stats(doc):
    word_counts = [len(sent) for sent in doc.sents]
    unique_words = [len(set([w.text for w in sent])) for sent in doc.sents]

    # Calculate the average number of unique words per sentence
    mean_unique_words = np.mean(unique_words)
    std_unique_words = np.std(unique_words)

    # Calculate the mean and standard deviation of word counts per sentence
    mean_word_count = np.mean(word_counts)
    stdev_word_count = np.std(word_counts)

    return mean_unique_words, std_unique_words, mean_word_count, stdev_word_count

def pos_tag_cnt_2(doc, tags = ['NN', 'NNS', 'NNP', 'NNPS', 
                                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                                'RB', 'RBR', 'RBS', 
                                'JJ', 'JJR', 'JJS',
                                'CC', 'CD', 'FW', 'MD', 'POS', 'PRP', 'PRP$', 
                                'WDT', 'WP', 'WP$', 'WRB']):
    tag_counts = Counter(word.tag_ for word in doc if word.tag_ in tags)
    return tag_counts

def c_at_1(scores, c_at_1_threshold = 0.05):
    out = []
    for score in scores:
        if abs(score - .5) < c_at_1_threshold:
            # Optimize c@1
            score = 0.5
        out.append(score)
    return out

def load_style_features(df):
    global nlp
    doc = []
    for i in df['text']:
        doc.append(nlp(i))

    df["doc"] = doc
    lex = df['text'].swifter.apply(lambda x: LexicalRichness(x))
    df['word_cnt'] = [i.words for i in lex]
    df['unique_word_cnt'] = [i.terms for i in lex]
    df['char_cnt'] = df['text'].str.len()
    df['avg_word_cnt'] = df['char_cnt']/df['word_cnt']
    df['quotation_cnt'] = df['text'].str.count('\"')
    df["unique_words_cnt"] = df['text'].swifter.apply(lambda x: len(Counter(re.sub(r'[^A-Za-z \n]', '', x).lower().split())))
    df["unique_words_relative"] = df["unique_words_cnt"] / df["word_cnt"]
    df['adverb_cnt'] = df['doc'].swifter.apply(count_adverbs) #15 min
    df['flesch_score'] = df['text'].swifter.apply(lambda x: textstat.flesch_reading_ease(x))
    df['hapax_rate'] = df['text'].swifter.apply(lambda x: hapax_legomenon_rate(x))
    df["stats"] = df["doc"].swifter.apply(get_sentence_stats)
    df[["unique_words_per_sentence_mean", "unique_words_per_sentence_stdev", "words_per_sentence_mean", "words_per_sentence_stdev"]] = pd.DataFrame(df["stats"].tolist(), index=df.index)

    # Drop the original 'stats' column
    df.drop(columns=["stats"], inplace=True)

    # The number of verbs not in the most common 5000 words
    all_synsets = list(wordnet.all_synsets())
    all_lemmas = [lemma.name() for synset in all_synsets for lemma in synset.lemmas()]
    fdist = FreqDist(all_lemmas)
    most_common_words = fdist.most_common(5000)
    most_common_words = [i[0] for i in most_common_words]
    
    def complex_verb_cnt(doc, most_common_words=most_common_words):
        verbs = [lemmatizer.lemmatize(word.text, pos='v') for word in doc if word.tag_.startswith('VB')]
        return len(verbs) - len(set(most_common_words) & set(verbs))

    lemmatizer = WordNetLemmatizer()
    df['complex_verb_cnt'] = df['doc'].swifter.apply(complex_verb_cnt) #8min
    df['sent_len_consistency'] = df["words_per_sentence_stdev"] / df["words_per_sentence_mean"]

    # POS TAG generation
    df['pos_counts'] = df['doc'].swifter.apply(pos_tag_cnt_2) #8min

    for tag in set().union(*df['pos_counts'].values):
        df[tag] = df['pos_counts'].apply(lambda x: x.get(tag, 0))

    # Drop the 'pos_counts' column if not needed
    df = df.drop('pos_counts', axis=1)

    # Contextual features
    extractor = ContextualRelevanceFeatures()
    def all_features(doc):
        res = {}
        features = extractor.extract_all_features(doc)
        for f in features:
            res.update(f)
        return res

    features = df['doc'].swifter.apply(all_features)
    df_feat = pd.DataFrame(list(features))
    df = pd.concat([df, df_feat], axis=1)
    del df_feat

    # Natural Text Pattern
    analyzer = NaturalTextPatternAnalyzer()

    def all_patterns(doc):
        res = {}
        results = analyzer.analyze_all_patterns(doc)
        for r in results:
            res.update(r)
        return res

    patterns = df['doc'].swifter.apply(all_patterns)
    df_patterns = pd.DataFrame(list(patterns))
    df = pd.concat([df, df_patterns], axis=1)

    del df_patterns

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction Script: PAN 2025 pindrop')
    parser.add_argument('-i', type=str,
                        help='Evaluaiton dir')
    parser.add_argument('-o', type=str, 
                        help='Output dir')
    args = parser.parse_args()

    # validate:
    if not args.i:
        raise ValueError('Eval dir path is required')
    if not args.o:
        raise ValueError('Output dir path is required')

    input_file = os.path.join(args.i, 'dataset.jsonl')
    output_file = os.path.join(args.o, 'output.jsonl')
    print("Writing answers to:", output_file , file=sys.stderr)

    df = pd.read_json(input_file, lines=True)
    df = load_style_features(df)

    # load model output
    summary_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ZOO['llama2-7b'],
        torch_dtype=torch.float16,
        device_map='auto'
    ).eval()
    summary_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ZOO['llama2-7b'], padding_side='left'
    )
    summary_tokenizer.pad_token = summary_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ZOO['gemma-2b'],
        torch_dtype=torch.float16,
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ZOO['gemma-2b'], padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # calculate biscope results
    results = []
    for text in df['text']:
        result = detect_single_sample(model, tokenizer, summary_model, summary_tokenizer, text, max_length=512, device=device)
        results.append(result)
        torch.cuda.empty_cache()

    bicap = np.array(results)
    feature_cols = [f'biscope_feature_{i}' for i in range(bicap.shape[1])]
    features_df = pd.DataFrame(bicap, columns=feature_cols)
    
    # Concatenate along columns
    X = pd.concat([df, features_df], axis=1)

    df_feature = X[SELECTED_FEATURES]
    clf = joblib.load('model.pkl')
    preds = clf.predict_proba(df_feature)[:,1]
    
    # run smoothing
    nomral_pred = c_at_1(preds)
    df['label'] = nomral_pred
    df[['id', 'label']].to_json(output_file, orient="records", lines=True, force_ascii=False)
    print('Job done', file=sys.stderr)
