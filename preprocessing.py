import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
le = LabelEncoder()

def preprocessing(data_path, imp_method, scale_method):
    # # read file(.arff) (read .arff, need to transfer 'byte' to 'float', got the 'float' value didn't match paper)
    # from scipy.io.arff import loadarff
    # d, meta = loadarff(data_path)
    # d = pd.DataFrame(d)
    # d = d.replace(b'?', np.nan)
    # # print(d.isnull().any())


    # read file(.csv)
    d = np.genfromtxt(data_path, delimiter=',')
    d[:, 13][d[:, 13] > 0] = 1 # label(Y) be 0 or 1
    d = pd.DataFrame(d)
    d.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    # print(d)

    # missing data
    inds = np.asarray(d.isnull()).nonzero()
    print('\nTreating missing data... ' + str(len(d.iloc[inds[0]])) + ' missing values')
    # print(d.iloc[inds[0]])
    print('imputation method:', imp_method)
    print('scale method: %s' % scale_method)
    d_new = imputation_scale(d, imp_method, scale_method)

    # split X, Y
    d = np.asarray(d_new)
    X = d[:, 0:13].astype(float)
    Y = d[:, 13]

    # Y to be 2-class
    encoder_Y = le.fit_transform(Y)
    Y = np_utils.to_categorical(encoder_Y)  # shape: (:, 2)

    return X, Y


def imputation_scale(data, imp_method, scale_method):
    cols = data.columns
    inds = np.asarray(data.isnull()).nonzero()
    var_name = cols[np.unique(inds[1])]
    data = pd.concat([data.drop(inds[0], axis=0), data.iloc[inds[0]]]).reset_index(drop=True) # rearrange NAN rows to the bottom
    inds = np.asarray(data.isnull()).nonzero()

    X1 = data[cols.drop(var_name)]
    # nom1 = [i for i in X1.columns if type(X1[i][0]) == bytes]
    # num1 = [i for i in X1.columns if type(X1[i][0]) == np.float64]
    # X1 = pd.concat([X1[nom1].apply(le.fit_transform), X1[num1]], axis=1)
    if scale_method == 'min_max': X1 = min_max_scale(X1)
    if scale_method == 'normalise': X1 = normalisation(X1)

    X2 = data[var_name]
    X3 = data[var_name].drop(inds[0], axis=0)
    # nom2 = [i for i in X2.columns if type(X2[i][0]) == bytes]
    # num2 = [i for i in X2.columns if type(X2[i][0]) == np.float64]
    # X3 = pd.concat([X3[nom2].apply(le.fit_transform), X3[num2]], axis=1)
    data = pd.concat([X1, X3], axis=1)

    if imp_method == 'x':
        data = data.drop(inds[0], axis=0)
    if imp_method == 'replace_mean':
        data = data.fillna(X3.mean())
    if imp_method == 'replace_med':
        data = data.fillna(X3.median())
    if imp_method == 'new_category':
        data = data.fillna(X3.max() + 1)

    if imp_method == 'MICE':
        Mice = MiceImputer()
        data = Mice.fit_transform(data)

    if imp_method == 'knn_1':
        for i in var_name:
            y_new = knn(X1.drop(inds[0], axis=0), X3[i], X1.iloc[inds[0]], i, 1)
            data.update(y_new)

    if imp_method == 'knn_3':
        for i in var_name:
            y_new = knn(X1.drop(inds[0], axis=0), X3[i], X1.iloc[inds[0]], i, 3)
            data.update(y_new)

    if scale_method == 'min_max': data = min_max_scale(data)
    if scale_method == 'normalise': data = normalisation(data)

    data = data[cols]
    return data

def min_max_scale(data):
    X = data[data.columns.drop('num')]
    X_scaled = (X-X.min())/(X.max()-X.min())
    X_scaled = pd.concat([X_scaled,data['num']],axis=1)
    return X_scaled
def normalisation(data):
    X = data[data.columns.drop('num')]
    X_normalised = (X-X.mean())/X.std(axis=0)
    X_normalised = pd.concat([X_normalised,data['num']],axis=1)
    return X_normalised

def knn(X1, y1, X2, var_name, k):
    X1 = X1[X1.columns.drop(['num'])]
    X2 = X2[X2.columns.drop(['num'])]

    clf = KNeighborsClassifier(k)
    clf.fit(X1, y1)
    y2 = clf.predict(X2)
    y_new = pd.DataFrame({var_name: np.append(np.asarray(y1), y2)})
    return y_new
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Imputer
class MiceImputer:
    model_dict_ = {}

    def __init__(self, seed_nulls=False, seed_strategy='mean'):
        self.seed_nulls = seed_nulls
        self.seed_strategy = seed_strategy

    def transform(self, X):
        col_order = X.columns
        new_X = []
        mutate_cols = list(self.model_dict_.keys())

        for i in mutate_cols:
            y = X[i]
            x_null = X[y.isnull()]
            y_null = y[y.isnull()].reset_index()['index']
            y_notnull = y[y.notnull()]

            model = self.model_dict_.get(i)

            if self.seed_nulls:
                x_null = model[1].transform(x_null)
            else:
                null_check = x_null.isnull().any()
                x_null = x_null[null_check.index[~null_check]]

            pred = pd.concat([pd.Series(model[0].predict(x_null)) \
                             .to_frame() \
                             .set_index(y_null), y_notnull], axis=0) \
                .rename(columns={0: i})

            new_X.append(pred)

        new_X.append(X[X.columns.difference(mutate_cols)])

        final = pd.concat(new_X, axis=1)[col_order]

        return final

    def fit(self, X):
        x = X.fillna(value=np.nan)

        null_check = x.isnull().any()
        null_data = x[null_check.index[null_check]]

        for i in null_data:
            y = null_data[i]
            y_notnull = y[y.notnull()]

            model_list = []
            if self.seed_nulls:
                imp = Imputer(strategy=self.seed_strategy)
                model_list.append(imp.fit(x))
                non_null_data = pd.DataFrame(imp.fit_transform(x))

            else:
                non_null_data = x[null_check.index[~null_check]]

            x_notnull = non_null_data[y.notnull()]

            if y.nunique() > 2:
                model = LinearRegression()
                model.fit(x_notnull, y_notnull)
                model_list.insert(0, model)
                self.model_dict_.update({i: model_list})
            else:
                model = LogisticRegression()
                model.fit(x_notnull, y_notnull)
                model_list.insert(0, model)
                self.model_dict_.update({i: model_list})

        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == '__main__':
    x, y = preprocessing('data/processed_data.csv', 'MICE', '')
    print(x[-3], y[-4])