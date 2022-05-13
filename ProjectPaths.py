#-*- coding: utf-8 -*-

import os
import pathlib

#######################
######## Paths ########
#######################

def make_log_file(filename:'str')->'str':
    '''
    Make Log file path string under <project_folder>/logs/filename
    And create the parent folder if it doesn't exist already 
    '''
    halfpath = os.path.join(os.path.dirname(os.getcwd()), 'logs')
    fullpath = os.path.join(halfpath, filename)
    file_parent = os.path.split(fullpath)[0]
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return fullpath

def make_data_path(filename:'str')->'str':
    '''
    Make Data file path string under <project_folder>/data/filename
    And create the parent folder if it doesn't exist already 
    '''
    halfpath=os.path.join(os.path.dirname(os.getcwd()), 'data')
    fullpath = os.path.join(halfpath, filename)
    file_parent = os.path.split(fullpath)[0]
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return fullpath

def make_keyword_path(filename:'str')->'str':
    '''
    Make Keyword file path string under <project_folder>/keyword/filename
    And create the parent folder if it doesn't exist already 
    '''
    halfpath = os.path.join(os.path.dirname(os.getcwd()), 'keywords')
    fullpath = os.path.join(halfpath, filename)
    file_parent = os.path.split(fullpath)[0]
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return fullpath

def make_model_path(filename:'str')->'str':
    '''
    Make Model file path string under <project_folder>/model/filename
    And create the parent folder if it doesn't exist already 
    '''
    halfpath= make_data_path('models')
    fullpath = os.path.join(halfpath, filename)
    file_parent = os.path.split(fullpath)[0]
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return fullpath

if __name__ == '__main__':
    pass
