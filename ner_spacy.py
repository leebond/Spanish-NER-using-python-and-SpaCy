from __future__ import unicode_literals, print_function

from nltk.corpus import conll2002
from spacy_ner_trainer import ner
import spacy

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

    TRAIN_DATA = process_sent(train_sents)
        
#    # training data
#    TRAIN_DATA = [
#        ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
#        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
#    ]

    # training model
#    model = ner.main(TRAIN_DATA)