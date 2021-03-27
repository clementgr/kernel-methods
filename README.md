# Kernel Methods
Data challenge: predicting whether a DNA sequence region is binding site to a specific transcription factor

# How to reproduce results?
To reproduce the model that gave my best score on the academic leaderboard, just run:
* `git clone https://github.com/clementgr/kernel-methods.git`
* `pip install -r requirements.txt`
* `python main.py --config config/best_submission.json` 

NB: it takes approx. 7min to run

Other config files allow to run different models:
* `python main.py --config config/lr.json`: Logistic Regression results on *_mat100 files
* `python main.py --config config/svm.json`: SVM results on *_mat100 files
* `python main.py --config config/rbf_krr.json`: Kernel Ridge Regression with RBF kernel on *_mat100 files
* `python main.py --config config/rbf_ksvm.json`: Kernel SVM with RBF kernel on *_mat100 files
* `python main.py --config config/spectrum.json`: Kernel SVM with spectrum kernel on DNA sequences 
* `python main.py --config config/mismatch.json`: Kernel SVM with mismatch kernel on DNA sequences 
