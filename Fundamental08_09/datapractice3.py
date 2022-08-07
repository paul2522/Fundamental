import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

fig = plt.figure()
csv_path = "flights.csv"
data = pd.read_csv(csv_path)
flights = pd.DataFrame(data)
#print(flights)

sns.barplot(data=flights, x='year', y='passengers')
plt.show()

sns.pointplot(data=flights, x='year', y='passengers')
plt.show()

sns.lineplot(data=flights, x='year', y='passengers')
plt.show()

sns.lineplot(data=flights, x='year', y='passengers', hue='month', palette='ch:.50')
plt.legend(bbox_to_anchor=(1.03, 1), loc=2) #legend 그래프 밖에 추가하기
plt.show()

sns.histplot(flights['passengers'])
plt.show()

#Heatmap

pivot = flights.pivot(index='year', columns='month', values='passengers')
print(pivot)

sns.heatmap(pivot)
plt.show()

sns.heatmap(pivot, linewidths=.2, annot=True, fmt="d")
plt.show()

sns.heatmap(pivot, cmap="YlGnBu")
plt.show()