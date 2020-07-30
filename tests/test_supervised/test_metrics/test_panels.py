# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_panels.py                                                   #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 29th 2020, 4:16:12 am                       #
# Last Modified : Wednesday, July 29th 2020, 4:16:12 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests panels, panel factory and repo."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories.algorithms import GradientDescent
from mlstudio.supervised.metrics.base import *
from mlstudio.supervised.metrics.regression import *
from mlstudio.supervised.metrics.panel import PanelRepo
from mlstudio.supervised.metrics.panel import RegressionPanel
from mlstudio.supervised.metrics.panel import BinaryClassPanel
from mlstudio.supervised.metrics.panel import MultiClassPanel
from mlstudio.supervised.metrics.panel import RegressionPanelFactory
from mlstudio.supervised.metrics.panel import BinaryClassPanelFactory
from mlstudio.supervised.metrics.panel import MultiClassPanelFactory
# --------------------------------------------------------------------------- #
@mark.panel
@mark.regression_panel
class RegressionPanelTests:

    def test_regression_panel_properties(self):
        panel = RegressionPanel(code='RP1',name="regression_panel_01",
                                description='First Regression Panel')
        assert panel.code == 'RP1', "Regression Panel: panel code error"
        assert panel.name == 'regression_panel_01', "Regression Panel: panel name error"
        assert panel.label == "Regression Panel 01", "Regression Panel: panel label error"
        assert panel.description == "First Regression Panel", "Regression Panel: panel description error"
        assert panel.panel_type == "Regression", "Regression Panel: panel type error"
        panel.description = "First Regression Panel Today"
        assert panel.description == "First Regression Panel Today", "Regression Panel: panel description error"

    def test_regression_panel_add_get_scorer(self):
        panel = RegressionPanel(code='RP2',name="regression_panel_02",
                                description='Second Regression Panel')
        panel.add_scorer(scorer=MeanSquaredError())
        assert isinstance(panel.get_scorer(key='MSE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_squared_error'), BasePerformance), "Regression Panel: add scorer error."
        with pytest.raises(KeyError):
            panel.add_scorer(scorer=MeanSquaredError())
        panel.add_scorer(scorer=MeanAbsolutePercentageError())
        assert isinstance(panel.get_scorer(key='MAPE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_absolute_percentage_error'), BasePerformance), "Regression Panel: add scorer error."


    def test_regression_panel_del_scorer(self):
        panel = RegressionPanel(code='RP3',name="regression_panel_03",
                                description='Third Regression Panel')
        panel.add_scorer(scorer=MeanSquaredError())
        panel.add_scorer(scorer=MeanAbsolutePercentageError())
        panel.del_scorer(key="MAPE")
        with pytest.raises(KeyError):
            panel.get_scorer(key="MAPE")
        with pytest.raises(KeyError):
            panel.get_scorer(key="mean_absolute_percentage_error")            
        assert isinstance(panel.get_scorer(key='MSE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_squared_error'), BasePerformance), "Regression Panel: add scorer error."        

    def test_regression_panel_print_scorers(self):
        panel = RegressionPanel(code='RP4',name="regression_panel_04",
                                description='Forth Regression Panel')
        panel.add_scorer(scorer=MeanSquaredError())
        panel.add_scorer(scorer=MeanAbsolutePercentageError())        
        panel.print_scorers()

    def test_regression_panel_call(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split
        est = GradientDescent.regression(epochs=5000)
        est.fit(X_train, y_train)
        panel = RegressionPanel(code='RP5',name="regression_panel_05",
                                description='Fifth Regression Panel') 
        panel.add_scorer(scorer=MeanSquaredError())
        panel.add_scorer(scorer=MeanAbsolutePercentageError())        
        panel.add_scorer(scorer=RootMeanSquaredError())                                               
        panel.add_scorer(scorer=NegativeMeanSquaredError())                                               
        panel.add_scorer(scorer=NegativeRootMeanSquaredError())                                               
        panel.add_scorer(scorer=R2())                                               
        panel.add_scorer(scorer=AdjustedR2())                                               
        panel.add_scorer(scorer=PercentVarianceExplained())     
        panel.add_scorer(scorer=MeanAbsoluteError())     
        panel.add_scorer(scorer=MeanSquaredLogError())     
        panel.add_scorer(scorer=MedianAbsoluteError())     
        df = panel(estimator=est, X=X_test, y=y_test, reference='regression_test_01')
        assert isinstance(df, pd.DataFrame), "Regression Panel: call error"
        print(df)

        




