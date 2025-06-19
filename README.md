AST
An Abstract Syntax Tree (AST) is a tree representation of the structural and semantic elements of a program's source code. It's a condensed version of a parse tree, focusing on the essential aspects of the code's structure. ASTs are used by compilers, interpreters, and other tools to analyze, manipulate, and generate code. 

Random Forest
It used multiple decision trees to make prediction,each tree is somewhat different and and the actuial classification is done by averaging the results
. It redices error and overfitting, which was the main issue in normal decision trees.

F1 Score
H.M of Precision and Recall

Precision: Ratio of Actual positives by total positives identified. 
True Positives / (True Positives + False Positives)

Recall :Ratio of identified positives by actual positives. 
True Positives / (True Positives + False Negatives)

Macro F1 Score
Mean of F1 scores

Cross-Validation (CV)
Splits training data in multiple parts, repeats training and testing on different parts (different splits) and averages the result. Makes sure your model isn’t just accidentally doing well on one lucky test split
Here, cv = 3 means model and training and tested 3 times

Confusion Matrix
Table used to describe the performance of a classification model.
Here, using Multiclass Classifier ( more than 2 ) namely (excellent, good, fair, poor)

​

Scalar fitting is done only on train data as test data is only supposed to be used for testing and not to be touched while training (can cause overfitting)

Download file from here!!
http://files.srl.inf.ethz.ch/data/py150.tar.gz

Model Link ( 1/ 10th size)
https://huggingface.co/Preygle/PyCritic/blob/main/code_eval_w_150k.pkl

Running model on kaggle for easier data loading (and not destroying my cpu)
