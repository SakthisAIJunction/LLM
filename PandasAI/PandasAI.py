# PandasAI - Pandas AI is a Python library that integrates generative artificial intelligence capabilities into Pandas, 
#            making dataframes conversational.It's not a replacement for pandas.

import os
import pandas as pd
import seaborn as sns

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

llm = OpenAI(model = "gpt-4")
pandas_ai = PandasAI(llm = llm)

titanic_data = sns.load_dataset('titanic')

# ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',   
#        'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
#        'alive', 'alone']

if __name__ == "__main__":
    resp = pandas_ai(titanic_data, prompt = "how many rows above the age 50?")
    print("response--->", resp)
    resp = pandas_ai(titanic_data, prompt = "which column has missing data")
    print("response--->", resp)
    resp = pandas_ai.generate_features(titanic_data)
    print("response--->", resp)
    resp = pandas_ai.clean_data(titanic_data)
    print("response--->", resp)
        pandas_ai.plot_histogram(titanic_data, column = 'sex')