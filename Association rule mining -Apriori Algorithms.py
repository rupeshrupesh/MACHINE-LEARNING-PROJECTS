# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:46:02 2019

@author: Rupesh S
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

data=pd.read_csv("D:\\MLRSMBAEX2-DataSet.csv", header=None)
data.head()
data.shape

records = []
for i in range(0, 9835):
    records.append([str(data.values[i,j]) for j in range(0, 32)])
    
item=[]
for i in records:
    item.append([j for j in i if (j!='nan')])
    

association_rules = apriori(item, min_support=0.0455, min_confidence=0.2, min_lift=1.1, min_length=6)
association_results = list(association_rules)

print(len(association_results))
print(association_results[0])


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

