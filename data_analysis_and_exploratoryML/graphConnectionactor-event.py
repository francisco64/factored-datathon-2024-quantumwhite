#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:57:56 2024

@author: franciscoreales
"""

import pandas as pd 
import numpy as np
import itertools

df=pd.read_csv('/Users/franciscoreales/Downloads/clusters.csv')

graph=[]

for i in range(1,11):
    featureColumn='feature_Centroid'+str(i)
    appereanceColumn='appereance_Centroid'+str(i)
    actors=[actor for actor in df[featureColumn] if not "UNITED_STATES" in actor and 'actor' in actor]
    
    aux_events=[event for event in df[featureColumn] if 'event' in event]
    aux_weights=[df[appereanceColumn][j]for j in range(0,len(df[featureColumn])) if 'event' in df[featureColumn][j]]
    event=aux_events[np.argmax(aux_weights)]
    
    combinations = list(itertools.permutations(actors, 2))

    # Convert the tuples to lists (optional)
    combinations = [list(pair)+[event]+["centroid_"+str(i)] for pair in combinations]
    
    graph+=combinations
    
graph=np.array(graph)

df=pd.DataFrame(data=graph)

df.to_csv('graph.csv',index=False)