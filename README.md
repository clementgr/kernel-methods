# Kernel Methods
Data challenge: predicting whether a DNA sequence region is binding site to a specific transcription factor<br>
Best score achieved using mismtach kernel, resulting in `0.64733` / `0.65200`  public / private score

# How to reproduce results?
To reproduce the model that gave my best score on the academic leaderboard, just run:
* `git clone https://github.com/clementgr/kernel-methods.git`
* `pip install -r requirements.txt`
* `python main.py --config config/best_submission.json` 

**NB:** it takes approx. 7min to reproduce scores using `best_submission.json`

Other config files allow to run different models (each takes less than 1min):
* `python main.py --config config/lr.json`: Logistic Regression on *_mat100 files
* `python main.py --config config/svm.json`: SVM on *_mat100 files
* `python main.py --config config/rbf_krr.json`: Kernel Ridge Regression with RBF kernel on *_mat100 files
* `python main.py --config config/rbf_ksvm.json`: Kernel SVM with RBF kernel on *_mat100 files
* `python main.py --config config/spectrum.json`: Kernel SVM with spectrum kernel on DNA sequences 
* `python main.py --config config/mismatch.json`: Kernel SVM with mismatch kernel on DNA sequences 
