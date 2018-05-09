# 2018 - IDSS PW3

Group 3

Implement **An integrated decision support system based on ANN and Fuzzy_AHP for heart failure risk prediction**.


## Intro
- [requirement.txt](requirement.txt) : list which packages used in this project.

- [setup.py](setup.py) : for packing up all.
- [main.py](main.py) : integrate all function and add user interface.
- [test.py](test.py) : integrate all function for testing each function.
- [preprocessing.py](preprocessing.py) : load data and deal with missing data.
- [faphy.py](faphy.py) : Fuzzy_AHP using pairwise_matrix to get the attribute's weights.
- [ann.py](ann.py) : train ANN to trained ANN for prediction.
- [eval.py](eval.py) : evaluate the model, using sensitivity/specificity, evaluation metrics, ROC and performance plot.

data/ :
  - processed_data.csv : [original dataset from UCI data repository](http://archive.ics.uci.edu/ml/datasets/heart+Disease).
  - weights : attribute's weights computed from Fuzzy_AHP.
  - ANNmodel.h5 : the ANN model information (including frame and weights).

image/ : store all images using in README.md
<br><br> 

## Result

- Programming Output<br>
  <img src="image/output.png" width="75%">
<br><br>
- Plot Curve<br>
  <img src="image/loss.png" width="45%">
  <img src="image/acc.png" width="45%">


## Evaluation

- evaluation metrics<br>
  <img src="image/confusion_matrix.png" width="65%">
<br><br>
- ROC
- performance plot


## Comparison
seed = 1, batch_size = 50, iteration=2000,
Using X normalization and attribute weight, 
ANN = 13-10-2
Test accuracy... **95.56%** (train: 84.13%)

- without weights without X normalization : 77.78%% (train: 80.95%)
- without X normalization : 91.11% (train: 83.73%)
- without weights : 91.11%% (train: 85.32%%)

- ANN = 13-10-1 : 93.33% (train: 83.73%)
- using kfold(=5) : 93.33% (train: 84.52%)

---
seed = 2, batch_size = 50, iteration=2000,
Using X normalization and attribute weight, 
ANN = 13-10-2
Test accuracy... **82.22%** (train: 85.71%)

- without weights without X normalization : 73.33%% (train: 87.30%)
- without X normalization : 80.00% (train: 86.11%)
- without weights : 84.44%% (train: 85.71%%)

- ANN = 13-10-1 : 82.22% (train: 84.92%)
- using kfold(=5) : 84.44% (train: 84.92%)