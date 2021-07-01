# External modules
import scipy
import numpy as np
from sklearn.utils import shuffle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import sklearn.datasets as skd
import openml

openml_ids = {'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
              'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
              'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
              'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
              'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
              'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
              'GesturePhaseSegmentationProcessed': 4538,
              'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
              'page-blocks': 30, 'yeast': 181, 'flags': 285,
              'visualizing_livestock': 685, 'diggle_table_a2': 694,
              'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477,
              'nursery': 1568, 'wall_robot_navigation':1497,'poker-hand':1569,
              'eye_movements':1044}
openml_ids_nans = {'heart-c': 49, 'dermatology': 35}


def load_dataset(dataset, n_samples=1000, n_features=10, n_classes=2,
                 seed=None):
    if dataset in list(openml_ids.keys()):
        dataset_id = openml_ids[dataset]
        data = openml.datasets.get_dataset(dataset_id)
        X, y, categorical, feature_names = data.get_data(
                                target=data.default_target_attribute,
                                )
        # TODO change NaN in categories for another category
        categorical_indices = np.where(categorical)[0]
        ct = ColumnTransformer([("Name_Of_Your_Step",
                                 OneHotEncoder(), categorical_indices)],
                               remainder="passthrough")
        X = ct.fit_transform(X)  # Categorical to binary
        n_samples = X.shape[0]           # Sample size
        n_features = X.shape[1]             # Data dimension
        # Assegurar que los valores en Y son correctos para todos los
        # resultados
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = y.max()+1      # Number of classes
    elif dataset == 'blobs':
        X, y = skd.make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=n_classes, cluster_std=1.0,
                              center_box=(-15.0, 15.0), shuffle=True,
                              random_state=seed)
    elif dataset == 'gauss_quantiles':
        X, y = skd.make_gaussian_quantiles(n_samples=n_samples,
                                           n_features=n_features,
                                           n_classes=n_classes,
                                           shuffle=True, random_state=seed)
    elif dataset == 'digits':
        X, y = skd.load_digits(n_class=n_classes, return_X_y=True)
        n_features = X.shape[0]             # Data dimension
    else:
        raise "Problem type unknown: {}"
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = si.fit_transform(X)
    if type(X) is scipy.sparse.csc.csc_matrix:
        X = X.todense()
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    X, y = shuffle(X, y, random_state=seed)

    # ## Report data used in the simulation
    print('----------------')
    print('Dataset description:')
    print('    Dataset name: {0}'.format(dataset))
    print('    Sample size: {0}'.format(n_samples))
    print('    Number of features: {0}'.format(n_features))
    print('    Number of classes: {0}'.format(n_classes))

    return X, y, n_classes, n_samples, n_features


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    print('Testing all datasets')
    for problem, dataset_id in openml_ids.items():
        print('Evaluating {}[{}] dataset'.format(problem, dataset_id))

        X, y, n_classes, n_samples, n_features = load_dataset(problem)

        lr = LogisticRegression()
        lr.fit(X, y)
        print('Logistic Regression score = {}'.format(lr.score(X, y)))
