# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \panel.py                                                         #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 29th 2020, 1:17:54 am                       #
# Last Modified : Wednesday, July 29th 2020, 1:17:54 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Metrics Panels: Collections of metrics"""
#%%
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
from datetime import date, time, datetime
import os
from pathlib import Path
import site
import warnings
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PERSISTENCE_DIR = os.path.join(DATA_DIR, "persistence")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
METRICS_PATH = os.path.join(REFERENCE_DIR, "metrics.xlsx")
site.addsitedir(PROJECT_DIR)
site.addsitedir(DATA_DIR)
site.addsitedir(METRICS_PATH)

import BTrees.OOBTree
import numpy as np
import pandas as pd
import persistent
from tabulate import tabulate
import transaction
import ZODB, ZODB.FileStorage


from mlstudio.data.persistence.database import PanelRepoDB
from mlstudio.utils.format import proper
from mlstudio.supervised.metrics.regression import *
from mlstudio.utils.validation import validate_regression_scorer
from mlstudio.utils.validation import validate_binaryclass_scorer
from mlstudio.utils.validation import validate_multiclass_scorer
# --------------------------------------------------------------------------- #
class AbstractPanel(persistent.Persistent):
    """Abstract base class for all Panel classes.

    Parameters
    ----------
    code : str
        A unique short 3 or 4 upper-case letter code for the panel.

    name : str
        A unique snake-case name for the panel.
    
    description: str
        A longer description of the purpose of the panel
    """
    #TODO: Add persistence
    def __init__(self, code, name=None, description=None):
        self._code = code
        self._name = name
        self._label = proper(name)
        self._description = description
        self._panel_type = "Abstract"
        self._scorers = OrderedDict()
        self._scores = None
        self._current_scores = None

    @property
    def code(self):
        """Property that returns the code of the panel."""
        return self._code

    @property
    def name(self):
        """Property that returns the name of the panel."""
        return self._name

    @property
    def label(self):
        """Returns name converted to proper case."""
        return self._label        

    @property
    def description(self):
        """Property that returns the description of the panel."""
        return self._description 

    @description.setter
    def description(self, x):
        self._description = x

    @property
    def panel_type(self):
        """Returns the type of the panel."""
        return self._panel_type

    @abstractmethod
    def _validate_scorer(self, scorer):
        """Confirms scorer is correct type."""
        pass

    def scores(self):
        return self._scores

    def current_scores(self):
        return self._current_scores

    def _search_scorer(self, key):
        """Searches by code and name."""
        try:
            return self._scorers[key]
        except:
            for scorer in self._scorers.values():
                if scorer.name == key:
                    return scorer
        return False

    def get_scorer(self, key):
        """Returns scorer based upon key
        
        Parameters
        ----------
        key : str
            This can be either the code or the name of the scorer.

        Returns
        -------
        scorer : Scorer object.

        Raises
        ------
        KeyError : If no scorer for the key is found.
        """

        scorer = self._search_scorer(key)      
        if scorer:
            return scorer
        else:
            msg = "Scorer key: {k} not found in this panel.".format(k=key)
            raise KeyError(msg)

    def add_scorer(self, scorer):
        """Adds scorer to the panel
        
        Parameters
        ----------
        scorer : A scorer object

        Raises
        ------
        KeyError : If the scorer already exists.
        """        
        if self._search_scorer(scorer.code):
            msg = "Scorer code {c} already exists in this panel.".format(c=scorer.code)
            raise KeyError(msg)
        self._scorers[scorer.code] = scorer        

    def del_scorer(self, key):
        """Deletes a scorer from the panel
        
        Parameters
        ----------
        key : str
            The code or name of the scorer

        Raises
        ------
        KeyError : If the scorer does not exist.
        """                
        scorer = self._search_scorer(key)
        if scorer:
            del self._scorers[scorer.code]
        else:
            msg = "No scorer exists with code or name equal to {k}.".format(k=key)
            raise ValueError(msg)

    def print_scores(self):
        """Prints scores from the last execution."""
        if self._scores:
            codes = []
            names = []
            labels = []
            scores = []
            for k, v in self._scores.items():
                names.append(k)
                codes.append(v.code)
                labels.append(v.label)
                scores.append(v.score)
            d = OrderedDict()
            d['Code'] = codes
            d['Name'] = names
            d['Label'] = labels
            d['Score'] = scores
            header = ['Name', 'Label', 'Score']
            df = pd.DataFrame(data=d)
            print(tabulate(df, headers=header, tablefmt="simple"))
        else:
            warnings.warn("No scores in this panel.")

    def print_scorers(self):
        """Prints existing scorers for the panel."""
        if self._scorers:
            print("\nPanel {p}: {d}".format(p=self._name, d=self._description))
            codes = []
            names = []
            labels = [] 
            for k, v in self._scorers.items():
                codes.append(k)
                names.append(v.name)
                labels.append(v.label)
            d = OrderedDict()
            d['Code'] = codes
            d['Name'] = names
            d['Label'] = labels
            header = ['Code', 'Name', 'Label']
            df = pd.DataFrame(data=d)
            print(tabulate(df, headers=header, tablefmt="simple"))
        else:
            warnings.warn("No scorers in this panel")


    def __call__(self, estimator, X, y, reference=None):
        """Iterates through the scorers and computes scores.
        
        Parameters
        ----------
        estimator : Estimator object
            The estimator to be evaluated

        X : array-like of shape (n_samples, n_features)
            The input data

        y : array-like of shape (n_samples,)         

        reference : str
            Snake cased key for referencing the current scoring pass
        
        """
        scores = OrderedDict()
        self._current_scores = None
        now = datetime.now()        
        scores['Date'] = date.today().strftime("%A %d. %B %Y")
        scores['Time'] = now.strftime("%H:%M:%S")
        scores['Reference'] = reference
        try:
            scores['Estimator'] = estimator.description
        except:
            scores['Estimator'] = estimator.__class__.__name__        
        #TODO: add logic that calls predict and score for
        for code, scorer in self._scorers.items():
            scores['Code'] = code
            scores['Name'] = scorer.name
            scores['Label'] = scorer.label
            scores['Score'] = estimator.score(X, y).copy()
            df = pd.DataFrame(data=scores, index=[0])
            if self._current_scores is not None:
                self._current_scores = pd.concat([self._current_scores, df])
            else:
                self._current_scores = df
            if self._scores is not None:
                self._scores = pd.concat([self._scores, df])
            else:
                self._scores = df
            
        return self._current_scores

# --------------------------------------------------------------------------- #
class RegressionPanel(AbstractPanel):
    """Regression metrics panel.

    Parameters
    ----------
    code : str
        A unique short 3 or 4 upper-case letter code for the panel.

    name : str
        A unique snake-case name for the panel.
    
    description: str
        A longer description of the purpose of the panel
    """
    def __init__(self, code, name=None, description=None):
        self._code = code
        self._name = name
        self._label = proper(name)
        self._description = description
        self._panel_type = "Regression"
        self._scorers = OrderedDict()
        self._scores = None
        self._current_scores = None

    def _validate_scorer(self, scorer):
        validation.validate_regression_scorer(scorer)

# --------------------------------------------------------------------------- #
class BinaryClassPanel(AbstractPanel):
    """Binary classification metrics panel.

    Parameters
    ----------
    code : str
        A unique short 3 or 4 upper-case letter code for the panel.

    name : str
        A unique snake-case name for the panel.
    
    description: str
        A longer description of the purpose of the panel
    """
    def __init__(self, code, name=None, description=None):
        self._code = code
        self._name = name
        self._label = proper(name)
        self._description = description
        self._panel_type = "Binary Classification"
        self._scorers = OrderedDict()
        self._scores = None
        self._current_scores = None

    def _validate_scorer(self, scorer):
        validation.validate_binaryclass_scorer(scorer)        

# --------------------------------------------------------------------------- #
class MultiClassPanel(AbstractPanel):
    """Multiclass classification metrics panel.

    Parameters
    ----------
    code : str
        A unique short 3 or 4 upper-case letter code for the panel.

    name : str
        A unique snake-case name for the panel.
    
    description: str
        A longer description of the purpose of the panel
    """
    def __init__(self, code, name=None, description=None):
        self._code = code
        self._name = name
        self._label = proper(name)
        self._description = description
        self._panel_type = "Binary Classification"
        self._scorers = OrderedDict()
        self._scores = None
        self._current_scores = None

    def _validate_scorer(self, scorer):
        validation.validate_multiclass_scorer(scorer)                
# --------------------------------------------------------------------------- #
class PanelRepo(persistent.Persistent):

    def __init__(self, name):
        self._name = name
        self._panels = {}
        self._db = PanelRepoDB()

    @property
    def name(self):
        return self._name

    def load(self):
        self._panels = self._db.load()

    def save(self):        
        self._db.save(self._panels)

    def add_panel(self, panel):
        if self._panels[panel.name]:
            msg = "Can't create panel. A panel with name {n} already exists.".format(n=panel.name)
            raise KeyError(msg)
        self._panels[panel.name] = panel
        self.save()

    def del_panel(self, name):
        if not self._panels[name]:
            msg = "Panel name {n} is not found.".format(n=name)
            raise LookupError(msg)
        del self._panels[name]
        self.save()

    def print(self):
        """Prints the list of panels in the repository."""
        print("{n} repository inventory".format(n=self._name))         
        names = []
        labels = []
        descriptions = []
        for name, panel in self._panels.items():
            names.append(name)
            labels.append(panel.label)
            descriptions.append(panel.description)
        d = OrderedDict()
        d['name'] = names
        d['label'] = labels
        d['desc'] = descriptions
        df = pd.DataFrame(data=d, index=[0])
        headers = ['Name', 'Label', 'Description']
        print(tabulate(df, headers=headers, tablefmt="simple"))

    def print_panel(self, name):
        """Prints a single panel."""
        if self._panels[name]:
            print("Panel: {n} {l} {d}".format(n=panel.name, l=panel.label, d=panel.description))
            self._panels[name].print_scorers()        


# --------------------------------------------------------------------------- #
class AbstractPanelFactory(ABC):
    """Abstract panel factory."""

    @abstractmethod
    def __init__(self, panel):        
        self._name = "abstract_panel_factory"
        self._label = "Abstract Panel Factory"
        self._description = "Abstract base class for panel factories."
        self._panel = panel
        self._metrics = None
        self._metrics_path = METRICS_PATH
        

    def __call__(self, name, label=None, description=None, scorers=None):
        
        panel = self._panel(name=name, label=label, description=description)
        for scorer in scorers:
            if isinstance(scorer, str):
                if self._metrics.isin([scorer]).any().any():
                    row = self._metrics.loc[self._metrics['name']==scorer]
                    if row['status'].values == 'done':
                        scorer = eval(row['classname'].values[0])
                        panel.add_scorer(scorer)
                    else:
                        msg = "Scorer {s} has not been implemented yet".format(s=scorer)
                        raise ValueError(msg)
                else:
                    msg = "Scorer {s} is not a valid scorer.".format(s=scorer)                   
                    raise ValueError(msg)                
            else:
                if scorer.name in self._metrics.values:
                    panel.add_scorer(scorer)
                else:
                    msg = "Scorer {s} is not a valid scorer.".format(s=scorer.name)                   
                    raise ValueError(msg)                    
        return panel

    def print_metrics(self):
        msg = "\n\n{l} \n{d} \nMetric Inventory\n".format(n=self._name, l=self._label, d=self._description)
        print(msg)

        print(self._metrics[['category','code', 'name', 'label']])

    @abstractmethod
    def load(self):
        """Loads the list of metrics and the metric/panel map."""
        pass

# --------------------------------------------------------------------------- #
class RegressionPanelFactory(AbstractPanelFactory):
    """Regression panel factory."""

    def __init__(self, panel):        
        self._name = "regression_panel_factory"
        self._label = "Regression Panel Factory"
        self._description = "Panel factory for panels containing regression metrics."
        self._panel = panel
        self._metrics = None
        self._metrics_path = METRICS_PATH    

    def load(self):
        """Loads the list of metrics and the metric/panel map."""
        xlsx = pd.ExcelFile(METRICS_PATH)
        self._metrics = pd.read_excel(xlsx, sheet_name='regression', header=0, usecols="A:H")

# --------------------------------------------------------------------------- #
class BinaryClassPanelFactory(AbstractPanelFactory):
    """Binary classification panel factory."""

    def __init__(self, panel):        
        self._name = "binaryclass_panel_factory"
        self._label = "Binary Classification Panel Factory"
        self._description = "Panel factory for panels containing binary classification metrics."
        self._panel = panel
        self._metrics = None
        self._metrics_path = METRICS_PATH    

    def load(self):
        """Loads the list of metrics and the metric/panel map."""
        xlsx = pd.ExcelFile(METRICS_PATH)
        self._metrics = pd.read_excel(xlsx, sheet_name='binary', header=0, usecols="A:H")

# --------------------------------------------------------------------------- #
class MultiClassPanelFactory(AbstractPanelFactory):
    """Multiclass classification panel factory."""

    def __init__(self, panel):        
        self._name = "multiclass_panel_factory"
        self._label = "Multiclass Classification Panel Factory"
        self._description = "Panel factory for panels containing multiclass classification metrics."
        self._panel = panel
        self._metrics = None
        self._metrics_path = METRICS_PATH    

    def load(self):
        """Loads the list of metrics and the metric/panel map."""
        xlsx = pd.ExcelFile(METRICS_PATH)
        self._metrics = pd.read_excel(xlsx, sheet_name='binary', header=0, usecols="A:H")

