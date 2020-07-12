# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \zodb.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Sunday, July 12th 2020, 5:10:36 am                          #
# Last Modified : Sunday, July 12th 2020, 5:10:36 am                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
#%%
import ZODB, ZODB.FileStorage

storage = ZODB.FileStorage.FileStorage('mlstudio.fs')
db = ZODB.DB(storage)
connection = db.open()
root = connection.root