import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import struct

#dataframe - df
df = pd.read_csv('https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day2/Icebreaking_dataset.csv')
df.head(31)

df = df.dropna()
names = df['Movie'].values
score = df['IMDB Score'].values
plt.scatter(names, score)
plt.xticks(names, rotation = 90)
plt.xlabel('Name of Movie')
plt.ylabel('IMBD Score')
plt.show()

#genre = df['Genre'].values
#modes = np.unique(genre)
#count = []
#for m in modes:
   # count.append(np.sum(genre == m))
#plt.bar(modes, count)
#plt.xlabel('Genre')
#plt.xticks(modes, rotation=75)
#plt.ylabel('Count')
#plt.show()


#min_score=np.min(score)
#max_score=np.max(score)
#bin_range=np.arange(min_score - 0.5,max_score + 0.5, 0.1)
#plt.hist(score,bin_range,ec='black')
#plt.xlabel('Score')
#plt.ylabel('Number of Movies')
#plt.show()


#print (np.mean(score))
#print (np.var(score))




