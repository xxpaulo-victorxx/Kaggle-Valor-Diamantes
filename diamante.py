# -*- coding: utf-8 -*-
"""
@author: PAULO
"""

import pandas as pd

df = pd.read_csv('../data/train.csv')
df.head()

df2 = pd.read_csv('../data/test.csv')
df.head()

# Adaptacao dos dados

df["cut"].unique()
df['clarity'].unique()
df['color'].unique()

# Criando Dicionário 
cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
clarity_dict = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
color_dict = {"J": 1,"I": 2,"H": 3,"G": 4,"F": 5,"E": 6,"D": 7}

df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
df = df.drop("id",axis=1) #Minha proposta

df.head()

df2['cut'] = df2['cut'].map(cut_class_dict)
df2['clarity'] = df2['clarity'].map(clarity_dict)
df2['color'] = df2['color'].map(color_dict)
df2 = df2.drop("id",axis=1) #Minha proposta
df2['price'] = 0

df2.head()

# Fim da Adaptacao dos dados

import sklearn
from sklearn import svm, preprocessing

# Embaralhar os dados seria necessário se os mesmos seguissem algum tipo de ordem.
# Poderia influenciar negativamente os resultados finais
#df = sklearn.utils.shuffle(df)

X = df.drop("price", axis=1).values
X = preprocessing.scale(X)
y = df['price'].values

X2 = df2.drop("price", axis=1).values
X2 = preprocessing.scale(X2)
y2 = df2['price'].values

X_train = X
y_train = y

X_test = X2
y_test = y2

clf = svm.SVR(kernel="rbf")
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))

# Criando arquivo texto com as predicoes
file_builder = open("predicted2.txt", "w+")
for X,y in zip(X_test, y_test):
    clf.predict([X])[0]= round((clf.predict([X])[0]),2)
    file_builder.write(f"{clf.predict([X])[0]} \n")
    print(f"Model:{clf.predict([X])[0]}")
    
file_builder.close()

# Mais tempo para rodar








