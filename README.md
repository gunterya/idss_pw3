# 2018 - IDSS PW3

Group 3

Reference: O.W. Samuel, G.M. Asogbon, A.K. Sangaiah, P. Fang, G. Li (2017) - An integrated decision support system based on ANN and Fuzzy_AHP for heart failure risK prediction. Expert Syst. Appl. 68, 163-172.


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


## Infrastructure


## Result

- Programming Output<br>
  <img src="image/output.png" width="75%">
<br><br>
- Plot Curve<br>
  <img src="image/loss.png" width="45%">
  <img src="image/acc.png" width="45%">


## Evaluation

(left: hybrid method, right: conventional ANN method) 

- evaluation metrics<br>
  <img src="image/confusion_matrix.png" width="45%">
  <img src="image/confusion_matrix_ann.png" width="45%">

- ROC<br>
  <img src="image/roc_curve.png" width="45%">
  <img src="image/roc_curve_ann.png" width="45%">

- performance plot<br>
  <img src="image/loss.png" width="45%">
  <img src="image/loss_ann.png" width="45%"><br>
    
    |            | hybrid method  | conventional ANN method  |
    |:----------:|---------------:|-------------------------:|
    | Train Accuracy |   84.13%      |  85.32%                    |
    | Test Accuracy  |   95.56%      |  91.11%                    |
    |Sensitivity |  100.00%       |  91.30%                     |
    |Specificity |  90.91%     |  90.91%                     |
    | FP rate   |  9.09%        |  9.09%                     |
    | FN rate   |  0.00%        |  8.70%                      |
    | Recall    |  100.00%         |  91.30%                    |
    | Precision |  92.00%       |  91.30%                    |
    | F1        |  95.83%       |  91.30%                    |


## Comparison

seed = 1, batch_size = 50, iteration=2000,<br>

| X(attribute) scale   |  missing data  | attribute weight  |  fix attribute w    |  ANN      |  test acc(train acc)  |
|:--------------------:|:--------------:|:-----------------:|:-------------------:|-----------|----------------------:|
| min-max              |  replace_mean  |  √               |  √                   | 13-10-2   |   84.78% (85.60%)     |
| normalization        |  replace_mean  |  √               |  √                   | 13-10-2   |   86.96% (85.60%)     |
| min-max              |  replace_med   |  √               |  √                   | 13-10-2   |   84.78% (85.21%)     |
| normalization        |  replace_med   |  √               |  √                   | 13-10-2   |   86.96% (85.60%)     |
| min-max              |  knn-1         |  √               |  √                   | 13-10-2   |   86.96% (84.44%)     |
| min-max              |  knn-3         |  √               |  √                   | 13-10-2   |   86.96% (84.44%)     |
| normalization        |  knn-1         |  √               |  √                   | 13-10-2   |   86.96% (85.21%)     |
| normalization        |  knn-3         |  √               |  √                   | 13-10-2   |   86.96% (85.21%)     |
| min-max              |  x             |  √               |  √                   | 13-10-2   |   **95.56%** (84.13%) |
| normalization        |  x             |  √               |  √                   | 13-10-2   |   93.33% (84.52%)     |
| x                    |  x             |  √               |  √                   | 13-10-2   |   91.11% (83.73%)     |
| min-max              |  x             |  x               |  √                   | 13-10-2   |   91.11% (85.32%)     |
| min-max              |  √             |  √               |  √                   | 13-10-1   |   93.33% (83.73%)     |


## TO-DO List

- [ ] No-fixed attribute weight
- [x] missing data treatment
- [x] Evaluation
- [x] Integrate 'missing data treatment'
    - [x] knn
    - [ ] MICE
- [ ] User Interface
- [ ] Pack project to .exe
- [ ] Word
- [ ] Slides
