Contents in this repository
Model file: nerooutput
Writeup: nlp_assignment4_writeup.pdf
Code: 1. ner_constrained.py
      2. ner_unconstrained.py
      3. ner_unconstrained_score.py
      4. spacy_ner_trainer.py
      5. conlleval.py
Result files: 1. constrained_results.txt
              2. unconst_results.txt

Usage
git clone all contents into the same folder

Constrained Version
run python script ner_constrained.py as per normal
> python ner_constrained.py
running the script will generate constrained_results.txt, the model is trained using training and development datasets to predict on the test datset.

Constrained Version
By taking a peek at the training and development data, I can tell that there seem to be latent
word features that can be captured. As such I developed the following word features below.
However, after testing the result on the dev set, I have reduced the list to a final set.
Word Features
1. Check if word is uppercase
2. Check if word is title case
3. Check if word is lowercase
4. Check if word is number
5. Check if word is not alphanumeric
6. Check if word is not alphabetical
7. Check if the length of the word is less than or equal to 2
8. Check if word contains dot
9. Check if word contains hyphen
Final Selected Word Features
1. Check if word is uppercase
2. Check if word is not alphabetical
3. Check if the length of the word is less than or equal to 2
However, not all word features resulted in better F1 performance, hence, I used a manual
forward and backward selection on the feature set. The final word features selected were base
on the overall F1 score they give on the dev set.
Window Features and Hyperparameters
1. Tokens within a window of 3 from target word
2. Pos tags within a window of 1 from target word
3. BIO tags within a window of 2 from target word
The above hyperparameters were selected base empirically base on the overall F1 on the dev
set.
Model
I have only used the Perceptron model because it is already achieve high F1 performance.
Hence, I went on further to experiment with the following hyperparameters,
a. Max_iter = 100
b. Early_stopping = True
c. Tol = 0.001
With that I am able to achieve the following results:
![alt text](https://raw.githubusercontent.com/leebond/Spanish-NER-using-python-and-SpaCy/sp_ner_const.png)

Unconstrained Version

Unconstrained Version
run python script ner_unconstrained.py per normal.
However, note that this file will import spacy_ner_trainer.py.
The runtime for this script will take 1.5hrs.
The script will save the model in ./neroutput. For convenience, I have added my own model in this submission.
Running the ner_unconstrained_score.py script will load the model (please changeinput name to ./neroutput_gh) and predict on the test dataset. And it will generate the unconst_result.txt file. 

To run the evaluation on the result files,
do 
> python conlleval.py <result file>
