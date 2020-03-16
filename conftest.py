#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : conftest.py                                                        #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 10:02:50 pm                         #
# Last Modified : Sunday, March 15th 2020, 10:02:50 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# %%
from pytest import fixture
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
# ============================================================================ #

@fixture(scope="session")
def split_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                            random_state=50)
    return X_train, X_test, y_train, y_test

@fixture(scope="session")
def get_generated_large_regression_data():
    X_path = "./tests/test_data/large_regression_X.npy"
    y_path = "./tests/test_data/large_regression_y.npy"
    if os.path.exists(X_path):
        X = np.load(file=X_path, allow_pickle=False)
        y = np.load(file=y_path, allow_pickle=False)
    else:
        X, y = datasets.make_regression(n_samples=1000000, 
                    n_features=100, bias=567,
                    effective_rank=20, noise=20)
        scaler = StandardScaler()    
        X = scaler.fit_transform(X)
        # Save numpy arrays as .npy files
        directory = os.path.dirname(X_path)
        X_file = os.path.basename(X_path)
        y_file = os.path.basename(y_path)
        save_numpy(X, directory, X_file)
        save_numpy(y, directory, y_file)        
    return X, y    

@fixture(scope='session')
def get_classes():
    c = Classes()
    classes = [LinearRegression(), LassoRegression(), RidgeRegression(),
               ElasticNetRegression(), LogisticRegression(), 
               MultinomialLogisticRegression()]
    for cls in classes:
        c.add_class(cls)
    return c

@fixture(scope="session")
def get_generated_medium_regression_data():
    X, y = datasets.make_regression(n_samples=1000, 
                n_features=10, bias=567,
                effective_rank=20, noise=20)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y    

@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_regression_hastie():    
    X, y = datasets.make_hastie_10_2()    
    scaler = StandardScaler()
    # pylint: disable=locally-disabled, no-member
    X = scaler.fit_transform(X)
    n_features = X.shape[1]    
    features = ['var_'+str(i) for i in range(n_features)]
    variables = features + ['TGT']
    y = np.atleast_2d(y).reshape(-1,1)
    a = np.concatenate((X, y), axis=1)
    df = pd.DataFrame(a, columns = variables)
    idx = np.random.random_integers(len(variables)-2)
    x = variables[idx]
    y = "var_4"
    z = "TGT"    

    return x, y, z, df

@fixture(scope="session")
def get_regression_data_df_plus():    
    boston = datasets.load_boston()    
    scaler = StandardScaler()
    # pylint: disable=locally-disabled, no-member
    X = scaler.fit_transform(boston.data)    
    X_df = pd.DataFrame(data=X, columns=boston.feature_names)
    y_np = boston.target
    y_series = pd.Series(y_np)
    y_df = pd.DataFrame(data=y_series, columns=['MEDV'])
    df = pd.concat([X_df, y_df], axis=1)
    # Randomly select a feature
    x_idx = np.random.random_integers(len(boston.feature_names)-1)
    y_idx = np.random.random_integers(len(boston.feature_names)-1)
    z_idx = np.random.random_integers(len(boston.feature_names)-1)
    x = boston.feature_names[x_idx]
    y = boston.feature_names[y_idx]
    z = boston.feature_names[z_idx]

    return x, y, z, df

@fixture(scope="session")
def get_binary_classification_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_multinomial_classification_data():
    X, y = datasets.load_iris(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_regression_data_w_validation(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test

@fixture(scope='session', params=[LinearRegression,
                                   LassoRegression,
                                   RidgeRegression,
                                   ElasticNetRegression])
def regression(request):
    return request.param

    
@fixture(scope='session')
def fit_multiple_models(get_regression_data):
        X, y = get_regression_data
        X = X[:,5]
        X = np.reshape(X, (-1,1))
        models = {}
        bgd = LinearRegression(epochs=200, seed=50)
        sgd = LinearRegression(epochs=200, seed=50, batch_size=1)
        mgd = LinearRegression(epochs=200, seed=50, batch_size=32)
        models= {'Batch Gradient Descent': bgd.fit(X,y),
                 'Stochastic Gradient Descent': sgd.fit(X,y),
                 'Mini-batch Gradient Descent': mgd.fit(X,y)}
        return models

@fixture(scope='session', params=['mae',
                                  'mse',
                                  'rmse',
                                  'mae',
                                  'r2',
                                  'var_explained',
                                  'nmse',
                                  'nrmse'])                                  
def models_by_metric(request):
    model = LinearRegression(metric=request.param)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)    
    return model        

@fixture(scope='session', params=['mae',
                                  'mse',
                                  'rmse',
                                  'medae'])                                  
def model_lower_is_better(request):
    model = LinearRegression(metric=request.param, early_stop=True,
                            val_size=0.3, precision=0.1,
                            patience=2)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)                            
    return model

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'nmse',
                                  'nrmse'])                                  
def model_higher_is_better(request):
    model = LinearRegression(metric=request.param, early_stop=True,
                            val_size=0.3, precision=0.1,
                            patience=2)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)                            
    return model

@fixture(scope='session', params=[TimeDecay(),
                                  StepDecay(),
                                  NaturalExponentialDecay(),
                                  ExponentialDecay(),
                                  InverseScaling(),
                                  PolynomialDecay(),
                                  Adaptive()])
def learning_rate_schedules(request):
    return request.param

@fixture(scope='session', params=['train_cost',
                                  'train_score',
                                  'val_cost',
                                  'val_score'])
def early_stop_monitor(request):
    return request.param

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'mae',
                                  'mse',
                                  'nmse',
                                  'rmse',
                                  'nrmse',
                                  'medae'])
def regression_metric(request):
    return request.param

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'nmse',
                                  'nrmse'])
def regression_metric_greater_is_better(request):
    return request.param

@fixture(scope='session', params=['mae',
                                  'mse',
                                  'rmse',                                  
                                  'medae'])
def regression_metric_lower_is_better(request):
    return request.param    

@fixture(scope='session', params=np.linspace(0.005,.01,5))
def get_learning_rate(request):
    return request.param

def least_squares(X,y):    
    X = np.insert(X, 0, 1, axis=1)
    w = np.linalg.lstsq(X,y)[0]
    return w

@fixture(scope='session')
def analytical_solution(get_regression_data):
    X, y = get_regression_data
    w = least_squares(X,y)    
    return w

@fixture(scope='session')
def analytical_solution_training_data(get_regression_data_w_validation):
    X_train, _, y_train, _ = get_regression_data_w_validation
    train_solution = least_squares(X_train, y_train)
    return train_solution


@fixture(scope='session')
def predict_y():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    gd = LinearRegression(epochs=5)
    gd.fit(X, y)
    y_pred = gd.predict(X)
    return y, y_pred

@fixture(scope='session')
def get_figure_path():
    return "tests/test_figures"

@fixture(scope='session')
def get_alpha():
    return 0.5


@fixture(scope='session')
def get_ratio():
    return 0.7


@fixture(scope='session')
def get_weights():
    return np.array([0.862485488, 0.113651277, 0.104906717, 0.213618469, 0.388680613,
                     0.98925317, 0.447474586, 0.935217135, 0.732583525, 0.076222466])


@fixture(scope='session')
def get_l1_cost():
    return 2.432046723


@fixture(scope='session')
def get_l1_grad():
    return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


@fixture(scope='session')
def get_l2_cost():
    return 1.780241816


@fixture(scope='session')
def get_l2_grad():
    return np.array([0.862485487970269,	0.113651277057817,	0.10490671672632,
                     0.213618469051441,	0.388680612571046,	0.989253169861413,
                     0.447474586387139,	0.935217134881623,	0.732583525054404,
                     0.0762224663412077])


@fixture(scope='session')
def get_elasticnet_cost():
    return 1.969468978


@fixture(scope='session')
def get_elasticnet_grad():
    return np.array([0.47937282319554,	0.367047691558673,	0.365736007508948,
                     0.382042770357716,	0.408302091885657,	0.498387975479212,
                     0.417121187958071,	0.490282570232243,	0.459887528758161,
                     0.361433369951181])


@fixture(scope='class')
def get_quadratic_y():
    return np.array([2257, 4744, 7040, 5488, 9755, 7435, 3812, 5296, 7300, 7041])


@fixture(scope='class')
def get_quadratic_y_pred():
    return np.array([8306, 6811, 1125, 4265, 1618, 3128, 2614, 2767, 3941, 4499])


@fixture(scope='class')
def get_quadratic_X():
    filename = "tests/test_operations/test_quadratic_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[2, 3, 4], skipfooter=2)
    X = df.values
    return X


@fixture(scope='class')
def get_quadratic_cost():
    return 9384127.6


@fixture(scope='class')
def get_quadratic_gradient():
    return np.array([-2109.4, -113092063.3, -34441317.9])


@fixture(scope='class')
def get_binary_cost_X():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[2, 3, 4])
    X = df.values
    return X


@fixture(scope='class')
def get_binary_cost_y():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0])
    y = df.values
    return y


@fixture(scope='class')
def get_binary_cost_y_pred():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[1])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_binary_cost():
    return 0.345424815


@fixture(scope='class')
def get_binary_cost_gradient():
    return np.array([-1.556243917, -960.1098781, -1342.758965])


@fixture(scope='class')
def get_categorical_cost_X():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[12, 13, 14])
    X = df.values
    return X


@fixture(scope='class')
def get_categorical_cost_y():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0, 1, 2])
    y = df.values
    return y


@fixture(scope='class')
def get_categorical_cost_y_pred():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[3, 4, 5])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_categorical_cost():
    return 0.367654163


@fixture(scope='class')
def get_categorical_cost_gradient():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[26, 27, 28], skipfooter=7)
    y_grad = df.values
    return y_grad


@fixture(scope='function')
def get_history():
    return History()


@fixture(scope='session', params=[[CanvasTitle(),LayoutTitle()],
                                  [CanvasLegend(),LayoutLegend()],
                                  [CanvasMargins(),LayoutMargins()],
                                  [CanvasSize(),LayoutSize()],
                                  [CanvasFont(),LayoutFont()],
                                  [CanvasColorBackground(),LayoutColorBackground()],
                                  [CanvasColorScale(),LayoutColorScale()],
                                  [CanvasColorAxisDomain(),LayoutColorAxisDomain()],
                                  [CanvasColorAxisScales(),LayoutColorAxisScales()],
                                  [CanvasColorAxisBarStyle(),LayoutColorAxisBarStyle()],
                                  [CanvasColorAxisBarPosition(),LayoutColorAxisBarPosition()],
                                  [CanvasColorAxisBarBoundary(),LayoutColorAxisBarBoundary()],
                                  [CanvasColorAxisBarTicks(),LayoutColorAxisBarTicks()],
                                  [CanvasColorAxisBarTickStyle(),LayoutColorAxisBarTickStyle()],
                                  [CanvasColorAxisBarTickFont(),LayoutColorAxisBarTickFont()],                            
                                  [CanvasColorAxisBarNumbers(),LayoutColorAxisBarNumbers()],
                                  [CanvasColorAxisBarTitle(),LayoutColorAxisBarTitle()]])

def canvas_layouts(request):
    return request.param

@fixture(scope='session')
def get_validation_rule_test_object():
    class TestClass:
        def __init__(self):
            self.b = True
            self.n = None
            self.i = 5
            self.d = '12/21/2001'
            self.f = 2.0
            self.color_hex = "#800000"
            self.color_rgb = "rgb(128,0,0)"
            self.e = ""
            self.s = "hats"
            self.sl = "aaron"
            self.sg = "zac"
            self.a_l = [4,3,2,5]
            self.a_n = [None, None, None]
            self.a_xn = [None, None, 9]
            self.a_g = [3,4,6,8]
            self.a_ge = [3,4,6,8]
            self.a_le = [5,6,2,9]
            self.a_b = np.array([True, False, True, True, False])
            self.a_i = [1,3,3,5,7,11,13,39]
            self.a_f = [1.5, 2.8, 3.9]
            self.a_s = ['apples','oranges', 'pears', 'bananas']
            self.a_sg = ['bananas','zulu', 'tomato', 'carrots']
            self.a_e = ["", "", ""]
            self.na_i = [[2, 44], 6]
            self.na_f = [[2.3, 4.4], 6.0]
            self.na_b = [[False, True], False]
            self.na_n = [[None, None], None]
            self.na_s = [["Discs", "Equalizer"], "Turntables"]
            self.na_xn = [[8, '9'], 'hat']
            self.na_e = [[66, "Hats", 2.0], [3, False, 3355]]
            self.na_ne = [['dub', "Hats", 2.0], [3, False, 55]]
    test_object = TestClass()
    return test_object

@fixture(scope='session')
def get_validation_rule_reference_object():
    class ReferenceClass:
        def __init__(self):
            self.b = False
            self.n = None
            self.i = 2
            self.d = "5/6/2017"
            self.f = 9.3
            self.e = ""
            self.s = "hats"
            self.a_l = [8,12,31,33]
            self.a_le = [5,6,2,9]
            self.a_g = [2,3,5,7]
            self.a_ge = [3,4,6,8]
            self.a_b = np.array([True, False, True, True, False])
            self.a_i = [1,2,3,5,7,9,11,13,15,17,19,21,39]
            self.a_f = np.linspace(1.0, 10,0, 30)
            self.a_s = ['apples','oranges']
            self.a_e = ["", "", ""]
            self.na_e = [[44, "Hats", 2.0], [3, False, 53]]
            self.na_ne = [['dac', "Hats", 2.0], [3, False, 32]]

    reference_object = ReferenceClass()
    return reference_object

# =========================================================================== #
#                                SKIPPED TESTS                                #
# =========================================================================== #
# BE CAREFUL WITH THIS. THIS CODE SKIPS ENTIRE DIRECTORIES
collect_ignore = ["setup.py"]
collect_ignore.append("tests/test_visualize/*.py")

