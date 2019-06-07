#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:29:59 2019

@author: macbook
"""
import spacy
from nltk.corpus import conll2002

 # load trained model
output_dir = "./neroutput"
model = spacy.load(output_dir)
def process_sent(dataset):
    OUTPUT_DATA = []
    
    for sentence in dataset:
        sent_temp = ""
        idxlen = 0
        entity_dict = {"entities": []}
        for sent in sentence:
            word, ner, iob = sent
            sent_temp += word + " "
            curr_idx = idxlen
            idxlen += len(word) + 1
            if iob != 'O':
                entity_dict["entities"].append((curr_idx, idxlen - 1, iob))
        OUTPUT_DATA.append((sent_temp, entity_dict))
    return OUTPUT_DATA
    

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    with open("unconst_results.txt", "w") as out:
        cnt = 0
        for sent in test_sents:
            for sent in sent:
#                token, pos, iob = sent
                doc = model(sent[0])
                if doc.ents != ():
                    tmp = [(ent.text, ent.label_) for ent in doc.ents]
                    word = tmp[0][0]
                    gold = sent[2]
                    pred = tmp[0][1]
                    out.write("{}\t{}\t{}\n".format(word,gold,pred))
                else:
                    word = sent[0]
                    gold = sent[2]
                    pred = 'O'
                    out.write("{}\t{}\t{}\n".format(word,gold,pred))
                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
            out.write("\n")

#print("Now run: python conlleval.py results.txt")    