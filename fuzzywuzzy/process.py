"""from fuzzywuzzy import process
import csv
with open("data.csv", 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        print(row)"""

"""from polyfuzz import PolyFuzz
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz.models import BaseMatcher


class MyModel(BaseMatcher):
    def match(self, from_list, to_list):
        # Calculate distances
        matches = [[fuzz.ratio(from_string, to_string) / 100 for to_string in to_list] 
                    for from_string in from_list]

        # Get best matches
        mappings = [to_list[index] for index in np.argmax(matches, axis=1)]
        scores = np.max(matches, axis=1)

        # Prepare dataframe
        matches = pd.DataFrame({'From': from_list,'To': mappings, 'Similarity': scores})
        return matches

custom_model = MyModel()
from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
#to_list = ["apple", "apples", "mouse"]
to_list = ["app"]


model = PolyFuzz(custom_model)
model.match(from_list, to_list)
print(model.get_matches("BERT"))"""


import pytest
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, RapidFuzz, BaseMatcher

from tests.utils import get_test_strings

'''
    Fuzzy based OFC 347 Data/String Similarity Matching
    1. Train the Model from the OFC347 Dataset of a Specific Column
       -- For Big Data, Samples can be pciked up randomnly
    2. Apply the PolyFuzzy Model with BERT 
    3. Pass the input data to find the similarity
    4. If the similarity value is one from the LIST
         -- Return as Best Match
       Else if Sort the best 10 Similarity values
         --  Find the mean
         -- If mean > .7 
            --- return as Good Match
       Else
        -- Doest not match (Not Similar)
'''

class HapletModel(BaseMatcher):
   def match(self, train_list, input_value):
        # Calculate distances
        matches = [[fuzz.ratio(from_string, input_value) / 100] for from_string in train_list]
           
        # Get best matches
        mappings = [input_value for index in np.argmax(matches, axis=1)]
        scores = np.max(matches, axis=1)

        # Prepare dataframe
        matches = pd.DataFrame({'From': from_list, 'To': mappings, 'Similarity': scores})
        matches = matches.sort_values(by=['Similarity'], ascending=False)
        return matches


def test_custom_model(from_list, to_string):
    custom_matcher = HapletModel()
    model = PolyFuzz(custom_matcher).match(from_list, to_string)
    matches = model.get_matches()
    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() >= 0.7
    #assert matches.Similarity.sort()

if __name__ == '__main__':
    custom_matcher = HapletModel()
    from_list = ['O342325', 'O463164', 'O998312', 'O923235', 'O759276', 'O707727', 'O965522', 'O915119', 'O486866', 'O465143', 'O226592', 'O479339']
    to_string = '428244'
    #from_list =['Baltimore', 'Boulder', 'San Leandro', 'Honolulu', 'Burnsville', 'High Point', 'Lynbrook', 'Portland', 'Beloit', 'Worcester', 'Miami', 'Erie', 'Mesquite', 'Tullahoma', 'Paterson', 'Homestead', 'New York']
    #from_list =['New']
    #to_string  = 'New York'
    #from_list=['N93827']
    #to_string='N93827'  

    # Sample IO
    model = PolyFuzz(custom_matcher).match(from_list, to_string)
    mat = model.get_matches("BERT")
    print(mat)
    #print(mat.Similarity.mean())