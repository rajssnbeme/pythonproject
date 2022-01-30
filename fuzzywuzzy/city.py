import pandas as pd
import numpy as np 


# method 1 
'''import geonamescache

gc = geonamescache.GeonamesCache()

countries = gc.get_countries()

cities = gc.get_cities()

import spacy 
#import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

doc = nlp('Baltimore, Boulder, Washington, CO, CA, HI, Bangor')

for ent in doc.ents:
    if ent.label_ == 'GPE':
        if ent.text in countries:
            print(f"Country : {ent.text}")
        elif ent.text in cities:
            print(f"City : {ent.text}")
        else:
            print(f"Other GPE : {ent.text}")
'''

# method 2
from polyfuzz import PolyFuzz

from rapidfuzz import fuzz
from polyfuzz.models import BaseMatcher
cities = pd.read_csv('us_cities_states.csv') 
# cities['City alias'] = cities['City alias'].apply(lambda x: str(x))
print(cities['City alias'] )

print("\n", type(cities))

df=pd.DataFrame(cities['City alias'])
vals = df.values
#print(vals)

T=vals.tolist()
#print(T)
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
# from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
from_list=T
#to_list = ["apple", "apples", "mouse"]
to_list = ["Holtsville"]


model = PolyFuzz(custom_model)
model.match(from_list, to_list)
print(model.get_matches("BERT"))

# GPE = Countries, cities, states.

'''count = 0
passed = 0
for i, city in enumerate(cities['City alias'].values):
    try:
        doc = nlp(city)
        for X in doc.ents: 
            if X.label_=='GPE': 
                count+=1
    except:
        passed +=1
        pass
    if i% 5000 == 0: print (i, count, passed)
print(f'Spacy knows {count} out of {cities.shape[0]}')
print('couldnt process:', passed)'''