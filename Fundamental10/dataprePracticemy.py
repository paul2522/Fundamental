import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = '/home/aiffel/Code/Practice3/vgsales.csv'
sales = pd.read_csv(csv_file_path)

# just print
# print(sales)
print(sales.head())