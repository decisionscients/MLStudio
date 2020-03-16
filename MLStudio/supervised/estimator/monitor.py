# %%
# =========================================================================== #
#                                MONITOR                                      #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \monitor.py                                                           #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday November 5th 2019, 8:47:45 pm                          #
# Last Modified: Saturday November 30th 2019, 10:37:20 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Module containing callbacks used to monitor and report training performance."""
import datetime
import numpy as np
import types
from mlstudio.supervised.estimator.callbacks import Callback

# --------------------------------------------------------------------------- #
#                             HISTORY CLASS                                   #
# --------------------------------------------------------------------------- #
class History(Callback):
    """Records history and metrics for training by epoch."""
    def on_train_begin(self, logs=None):
        """Sets instance variables at the beginning of training.
        
        Parameters
        ----------
        logs : Dict
            Dictionary containing the X and y data
        """ 
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now() 
        self.epoch_log = {}
        self.batch_log = {}        

    def on_train_end(self, logs=None):        
        """Sets instance variables at end of training.
        
        Parameters
        ----------
        logs : Dict
            Not used 
        """
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 

    def on_batch_end(self, batch, logs=None):
        """Updates data and statistics relevant to the training batch.
        
        Parameters
        ----------
        batch : int
            The current training batch
        
        logs : dict
            Dictionary containing batch statistics, such as batch size, current
            weights and training cost.
        """
        self.total_batches = batch
        for k,v in logs.items():
            self.batch_log.setdefault(k,[]).append(v)        

    def on_epoch_end(self, epoch, logs=None):
        """Updates data and statistics relevant to the training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch
        
        logs : dict
            Dictionary containing data and statistics for the current epoch,
            such as weights, costs, and optional validation set statistics
            beginning with 'val_'.
        """
        logs = logs or {}
        self.total_epochs = epoch
        for k,v in logs.items():
            self.epoch_log.setdefault(k,[]).append(v)

# --------------------------------------------------------------------------- #
#                            PROGRESS CLASS                                   #
# --------------------------------------------------------------------------- #              
class Progress(Callback):
    """Class that reports progress at designated points during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Reports progress at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch

        logs : Dict
            Statistics obtained at end of epoch
        """
        if self.model.verbose and (epoch % self.model.checkpoint == 0):
            items_to_report = ('epoch', 'train', 'val')
            logs = {k:v for k,v in logs.items() if k.startswith(items_to_report)}
            progress = "".join(str(key) + ': ' + str(round(value,4)) + ' ' \
                for key, value in logs.items())
            print(progress)

# --------------------------------------------------------------------------- #
#                                SUMMARY                                      #
# --------------------------------------------------------------------------- #

center = 25

def summary(history):
    """Summarizes statistics for model.

    Parameters
    ----------
    history : history object
        history object containing data and statistics from training.
    """
    metric = history.params.get('metric', "")

    print("\nOptimization Summary")
    print("                     Name: " + history.params.get('name'))
    print("                    Start: " + str(history.start))
    print("                      End: " + str(history.end))
    print("                 Duration: " + str(history.duration) + " seconds.")
    print("                   Epochs: " + str(history.total_epochs))
    print("                  Batches: " + str(history.total_batches))
    print("\n")
    print("Performance Summary")
    print("      Final Training Loss: " +
          str(round(history.epoch_log.get('train_cost')[-1],4)))
    if history.epoch_log.get('train_score'):
        print("     Final Training Score: " +\
             str(round(history.epoch_log.get('train_score')[-1],4))
              + " " + history.params.get('metric'))
    if history.epoch_log.get('val_cost'):
        print("    Final Validation Loss: " +
              str(round(history.epoch_log.get('val_cost')[-1],4)))
    if history.epoch_log.get('val_score'):
        print("   Final Validation Score: " + \
            str(round(history.epoch_log.get('val_score')[-1],4))
                + " " + metric)
    if history.epoch_log.get('theta')[-1].shape[0] < 10:
        print("          Final Intercept: " + str(history.epoch_log.get('theta')[-1][0]))
        print("       Final Coefficients:\t " +\
            str(history.epoch_log.get('theta')[-1][1:]).replace('\n','\n\t\t\t'))
    print("\nModel Parameters")

    for p, v in history.params.items():
        label_length = len(p)
        spaces = center - label_length
        if isinstance(v, (str, bool, int, list, np.ndarray, types.FunctionType, float)) \
                or v is None:
            p = " " * spaces + p + ": "
            # If v is a function type, it is the lambda function that
            # initializes the regularizer to zeros if the parameter
            # is None. If this is the case, we'll print "None" for
            # this parameter.
            if isinstance(v, types.FunctionType):
                v = ""
            print(p + str(v))
        else:
            _recur(p, v)

    print("\nData")
    print("        Num. Observations: " +\
             str(history.model.X.shape[0]))
    print("            Num. Features: " +\
             str(history.model.X.shape[1]-1))


def _recur(callable_type, callable_object):
    callable_name = callable_object.name
    spaces = center - len(callable_type)
    callable_type = " " * spaces + callable_type + ":"
    print(callable_type, callable_name)
    config = callable_object.get_params()
    if len(config) > 0:
        for k, v in config.items():
            if isinstance(v, (str, bool, int, list, np.ndarray, types.FunctionType, float)) \
                    or v is None:
                spaces = center - len(k)
                k = " " * spaces + k + ": "
                # If v is a function type, it is the lambda function that
                # initializes the regularizer to zeros if the parameter
                # is None. If this is the case, we'll print "None" for
                # this parameter.
                if isinstance(v, types.FunctionType):
                    v = None
                print(k + str(v))
            else:
                _recur(k, v)

        
