Contents in this zip
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
Unzip all contents into the same folder

Constrained Version
run python script ner_constrained.py as per normal
> python ner_constrained.py
running the script will generate constrained_results.txt, the model is trained using training and development datasets to predict on the test datset.

Unconstrained Version
run python script ner_unconstrained.py per normal.
However, note that this file will import spacy_ner_trainer.py.
The runtime for this script will take 1.5hrs.
The script will save the model in ./neroutput. For convenience, I have added my own model in this submission.
Running the ner_unconstrained_score.py script will load the model (please changeinput name to ./neroutput_gh) and predict on the test dataset. And it will generate the unconst_result.txt file. 

To run the evaluation on the result files,
do 
> python conlleval.py <result file>
