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

results/ : store all images using in README.md
<br><br> 


## Infrastructure


## Result
(BEST ONE: without missing data treatment, min_max scale X, with attribute weights)

- [Programming Output](results/min_max-x-1/out.txt)
<br><br>
- Plot Curve<br>
  <img src="results/min_max-x-1/loss.png" width="45%">
  <img src="results/min_max-x-1/acc.png" width="45%">
  
  
## Evaluation

(left: hybrid method, right: conventional ANN method)

- evaluation metrics<br>
  <img src="results/min_max-x-1/confusion_matrix.png" width="45%">
  <img src="results/min_max-x-0/confusion_matrix.png" width="45%">

- ROC<br>
  <img src="results/min_max-x-1/roc_curve.png" width="45%">
  <img src="results/min_max-x-0/roc_curve.png" width="45%">

- performance plot<br>
  <img src="results/min_max-x-1/loss.png" width="45%">
  <img src="results/min_max-x-0/loss.png" width="45%"><br>

    |            | [hybrid method](results/min_max-x-1/out.txt)  | [conventional ANN method](results/min_max-x-0/out.txt)  |
    |:----------:|---------------:|-------------------------:|
    | Train Accuracy |   84.13%      |  85.32%                    |
    | Test Accuracy  |   95.56%      |  91.11%                    |
    |Sensitivity |  100.00%       |  91.30%                     |
    |Specificity |  90.91%     |  90.91%                     |
    | FP rate   |  9.09%        |  9.09%                     |
    | FN rate   |  0.00%        |   8.70%                      |
    | Recall    |  100.00%         |  91.30%                    |
    | Precision |  92.00%       |  91.30%                    |
    | F1        |  95.83%       |  91.30%                    |
    
    
## Comparison

seed = 1, batch_size = 50, iteration=2000,<br> 

| X(attribute) scale   |  missing data  | attribute weight  |  fix attribute w    |  ANN      |  test acc(train acc)  |
|:--------------------:|:--------------:|:-----------------:|:-------------------:|-----------|----------------------:|
|min_max|replace_mean|1| √ | 13-10-2|80.43(86.38)|
|normalise|replace_mean|1| √ | 13-10-2|82.61(86.38)|
|min_max|replace_med|1| √ | 13-10-2|78.26(85.21)|
|normalise|replace_med|1| √ | 13-10-2|80.43(85.6)|
|MICE|replace_med|1| √ | 13-10-2|82.61(86.38)|
|MICE|replace_med|1| √ | 13-10-2|82.61(86.38)|
|min_max|knn_1|1| √ | 13-10-2|78.26(85.6)|
|normalise|knn_1|1| √ | 13-10-2|80.43(86.38)|
|min_max|knn_3|1| √ | 13-10-2|78.26(85.99)|
|normalise|knn_3|1| √ | 13-10-2|78.26(85.99)|
|min_max|x|1| √ | 13-10-2|**95.56**(83.73)|
|normalise|x|1| √ | 13-10-2|93.33(84.13)|
|min_max|x|0| √ | 13-10-2|**91.11**(85.71)|


## TO-DO List

- [ ] No-fixed attribute weight
- [x] missing data treatment
- [x] Evaluation
- [x] Integrate 'missing data treatment'
- [ ] User Interface
- [ ] Pack project to .exe
- [ ] [Word](https://docs.google.com/document/d/1eVly1WEBN5DUt3R2okgRJKMU7RMSDm8R6JkDb2FgmsM/edit)
- [ ] Slides
