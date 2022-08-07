import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.figure
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

csv_path = os.getenv("HOME") +"/Code/AiffelPractice/Practice7/data/Pokemon.csv"
original_data = pd.read_csv(csv_path)

# 기존 데이터에서 copy 해서 poket
pokemon = original_data.copy() # DataFrame의 copy는 기본적으로 deep copy 이다.
# print("pokemon")
# print(pokemon.head())
# print(pokemon.shape)

legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)
# print("legendary")
# print(legendary.head())
# print(legendary.shape)

ordinary = pokemon[pokemon["Legendary"] == False].reset_index(drop=True)
# print("ordinary")
# print(ordinary.head())
# print(ordinary.shape)

# print("null sum")
# print(pokemon.isnull().sum()) # Type2 : 386

# print("pokemon.columns")
# print(len(pokemon.columns)) # 13
# print(pokemon.columns)

# # set는 집합 set를 나타낸다. 집합 특징은 중복 제거하고 나오는거
# print("tyeps of #(id) number in pokemon")
# print(len(set(pokemon["#"])))
# # 총 데이터 개수는 800개인데 721로 나옴 중복되는게 있네
# print("pokemon[#] = 6")
# print(pokemon[pokemon["#"] == 6])

# # 포켓몬 이름 중복제거하거 나오는 거 개수
# print("pokemon[Name] 개수")
# print(len(set(pokemon["Name"])))
# # 800이 나오므로 이름은 unique 한데 #(id)는 그렇지 않다는 걸 알 수 있습니다.

# print("6, 10 행 출력")
# print(pokemon.loc[[6, 10]])

# print("Type1을 세트화한걸 리스트한 길이, Type2를 세트화한걸 리스트한 길이")
# print(len(set(pokemon["Type 1"])),len(set(pokemon["Type 2"])))
# print(len(list(set(pokemon["Type 1"]))), len(list(set(pokemon["Type 2"]))))
# # 왜 굳이 list를 취해서 len을 구하지?????

# print("Type2 세트에서 Type1 세트를 차집합 해봅시다.")
# print(set(pokemon["Type 2"]) - set(pokemon["Type 1"]))
# nan이 나옵니다. 즉 Type2는 최소한 Type1에 포함되는 거네요

# print("Type1을 세트화해서 list로 만든 뒤에 len 값을 구하고 출력도 해봅시다.")
types = list(set(pokemon["Type 1"]))
types2 = set(pokemon["Type 1"])
# print(len(types))
# print(types)
# print("set 상태에서 똑같이 출력해봅니다")
# print(len(types2))
# print(types2)

# print("Type2의 결측값의 개수를 구해봅시다.")
# print(pokemon["Type 2"].isna().sum())
#isna() 는 결측값이냐? True, False로 나오고 이 개수를 sum한다.

# plt.figure('pokemon by Type1', figsize=(13, 7))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

# plt.subplot(211) # 추가 플롯 생성
# sns.countplot(data=ordinary, x="Type 1", order=types2).set_xlabel('')
# # Type1 에 따라 ordinary pokemon들을 countplot 해봅시다.
# plt.title("[Ordinary Pokemons]")

# plt.subplot(212)
# sns.countplot(data=legendary, x="Type 1", order=types2).set_xlabel('')
# # Type1 에 따라 legendary pokemon들을 countplot 해봅시다.
# plt.title("[Legendary Pokemons]")
# plt.show()

# pivot_table로 퍼센트 계산 가능
# pivot1 = pd.pivot_table(pokemon, index="Type 1", values="Legendary").sort_values(by=["Legendary"], ascending=False)
# print(pivot1)

# plt.figure(figsize=(12, 10))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

# plt.subplot(211)
# sns.countplot(data=ordinary, x="Type 2", order=types).set_xlabel('')
# plt.title("[Ordinary Pokemons]")

# plt.subplot(212)
# sns.countplot(data=legendary, x="Type 2", order=types).set_xlabel('')
# plt.title("[Legendary Pokemons]")

# plt.show()

# pivot2 = pd.pivot_table(pokemon, index="Type 2", values="Legendary").sort_values(by=["Legendary"], ascending=False)
# print(pivot2)

# 스탯 분석
# stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
# print(stats)

# 스탯들의 합계가 total값과 일치 하는지 확인해봅시다.
# print("#0 pokemon: ", pokemon.loc[0, "Name"])
# print("total: ", int(pokemon.loc[0, "Total"]))
# print("stats: ", list(pokemon.loc[0, stats]))
# print("sum of all stats: ", sum(list(pokemon.loc[0, stats])))\

# 위와 같이 일치하는 포켓몬의 개수는 전체 데이터수 800과 동일하다
# print(sum(pokemon['Total'].values == pokemon[stats].values.sum(axis=1)))

# fig, ax = plt.subplots()
# fig : 전체 subplot, ax는 전체 중 낱낱개
# fig.set_size_inches(12, 6)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

# Type1에 따라 Total 점을 찍는데 Legendary 에 따라 색을 구분한다.
# sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
# plt.show()

# 그래프 6개를 3x2 형식으로 그립니다.
# figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
# figure.set_size_inches(12, 18)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

# sns.scatterplot(data=pokemon, y="Total", x="HP", hue="Legendary", ax=ax1)
# sns.scatterplot(data=pokemon, y="Total", x="Attack", hue="Legendary", ax=ax2)
# sns.scatterplot(data=pokemon, y="Total", x="Defense", hue="Legendary", ax=ax3)
# sns.scatterplot(data=pokemon, y="Total", x="Sp. Atk", hue="Legendary", ax=ax4)
# sns.scatterplot(data=pokemon, y="Total", x="Sp. Def", hue="Legendary", ax=ax5)
# sns.scatterplot(data=pokemon, y="Total", x="Speed", hue="Legendary", ax=ax6)
# plt.show()

# plt.figure(figsize=(12, 10))   # 화면 해상도에 따라 그래프 크기를 조정해 주세요.
# plt.subplot(211) # 2행 1열 짜리 공간에서 1번째
# sns.countplot(data=ordinary, x="Generation").set_xlabel('')
# plt.title("[All Pkemons]")
# plt.subplot(212) # 2행 1열 짜리 공간에서 1번째
# sns.countplot(data=legendary, x="Generation").set_xlabel('')
# plt.title("[Legendary Pkemons]")
# plt.show()

# fig, ax = plt.subplots()
# fig.set_size_inches(8, 4)

# sns.scatterplot(data=legendary, y="Type 1", x="Total")
# plt.show()

# 전설의 포켓몬이 가지는 Total을 세트화 후 리스트 해서 출력
# print(sorted(list(set(legendary["Total"]))))

# fig, ax = plt.subplots()
# fig.set_size_inches(8, 4)

# sns.countplot(data=legendary, x="Total")
# plt.show()

# print(round(65 / 9, 2))
# 보통 포켓몬의 Total을 세트화 하고 리스트해서 출력하고 개수 세기
# print(sorted(list(set(ordinary["Total"]))))
# print(len(sorted(list(set(ordinary["Total"])))))
# print(round(735 / 195, 2))

# 전설 포켓몬 저렇게 자른뒤에 다 사슬연결해서 데이터프레임으로 만들고 출력
n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
# # print(names)

# # name DataFrame의 13행부터 22행까지를 formes
# formes = names[13:23]
# # print(formes)

# # 람다 사용하기 apply : 원하는 내용 적용해서 분리해서 볼때
legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))    
# print(legendary.head())
ordinary["name_count"] = ordinary["Name"].apply(lambda i: len(i))    
# print(ordinary.head())

# # 2행 1열 그래프 만들고 namecount(이름 길이)에 따라 그래프 만들기
# plt.figure(figsize=(12, 10))   # 화면 해상도에 따라 그래프 크기를 조정해 주세요.
# plt.subplot(211)
# sns.countplot(data=legendary, x="name_count").set_xlabel('')
# plt.title("Legendary")
# plt.subplot(212)
# sns.countplot(data=ordinary, x="name_count").set_xlabel('')
# plt.title("Ordinary")
# plt.show()

# print(round(len(legendary[legendary["name_count"] > 9]) / len(legendary) * 100, 2), "%")
# print(round(len(ordinary[ordinary["name_count"] > 9]) / len(ordinary) * 100, 2), "%")

# pokemon 데이터 프레임에 name_count 라는 이름 길이의 열을 만든다.
pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
# print(pokemon.head())

# 이름 길이가 10이상인 경우에 True인 long_name 열을 만든다.
pokemon["long_name"] = pokemon["name_count"] >= 10
# print(pokemon.head())

# isalpha() : 알파벳이 아닌 문자가 들어간 경우 처리하기

# 이름에서 빈칸 제거한 열 생성
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))

# 빈칸 제거한 열에서 다 알파벳만 있는지 확인하기
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())

# name_isalpha 가 false 인 경우의 pokemon 데이터프레임 형태 및 전체 출력
# print(pokemon[pokemon["name_isalpha"] == False].shape)
# print(pokemon[pokemon["name_isalpha"] == False])

# 알파벳 아닌거 바꿔주기
pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

# print(pokemon.loc[[34, 37, 90, 131, 252, 270, 487, 525, 794]])

pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
# print(pokemon[pokemon["name_isalpha"] == False])

import re

name = "CharizardMega Charizard X"
name_split = name.split(" ")
# print(name_split)
temp = name_split[0]
# print(temp)
# 글자형식이 대문자+소문자 형식을에서 찾는다. temp 에서
# CharizardMega = Charizard + Mega
tokens = re.findall('[A-Z][a-z]*', temp)
# print(tokens)

tokens = []
for part_name in name_split:
    a = re.findall('[A-Z][a-z]*', part_name)
    tokens.extend(a)
tokens

def tokenize(name):
    name_split = name.split(" ")
    
    tokens = []
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a) #list 에 a 추가
        
    return np.array(tokens)

name = "CharizardMega Charizard X"
# print(tokenize(name))

# 전설 포켓몬 이름들에서 토큰화해서 나온 값들로 이루어진 리스트 만들기
all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

# print(len(set(token_set)))
# print(token_set)

from collections import Counter
a = [1, 1, 0, 0, 0, 1, 1, 2, 3]
Counter(a)

Counter(a).most_common()

# 가장 많이 사용된 토큰 10개
most_common = Counter(token_set).most_common(10)
# print(most_common)

# 이름 중에 해당 토큰이 포함되어있는 경우
for token, _ in most_common:
    # pokemon[token] = ... 형식으로 사용하면 뒤에서 warning이 발생합니다
    pokemon[f"{token}"] = pokemon["Name"].str.contains(token)

# print(pokemon.head(10))

# print(types)

# Type1, Type2 의 특징마다 열을 만들고 True/False 로 생성하기
for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
    
pokemon[[["Type 1", "Type 2"] + types][0]].head()

# print(original_data.shape)
original_data.head()

original_data.columns
# print(original_data.columns)

features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
target = 'Legendary'

X = original_data[features]
# print(X.shape)
X.head()

y = original_data[target]
# print(y.shape)
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=25)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# print(len(pokemon.columns))
# print(pokemon.columns)

features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 
            'name_count', 'long_name', 'Forme', 'Mega', 'Mewtwo', 'Kyurem', 'Deoxys', 'Hoopa', 
            'Latias', 'Latios', 'Kyogre', 'Groudon', 'Poison', 'Water', 'Steel', 'Grass', 
            'Bug', 'Normal', 'Fire', 'Fighting', 'Electric', 'Psychic', 'Ghost', 'Ice', 
            'Rock', 'Dark', 'Flying', 'Ground', 'Dragon', 'Fairy']

# print(len(features))

target = "Legendary"
target

# 추려진 feature로 만든 X 데이터
X = pokemon[features]
# print(X.shape)
X.head()
# 정답지가 되는 y 데이터
y = pokemon[target]
# print(y.shape)
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

model = DecisionTreeClassifier(random_state=25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))

# print(classification_report(y_test, y_pred))

for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
    
print(pokemon[[["Type 1", "Type 2"] + types][0]].head())
print(pokemon[["Type 1", "Type 2"] + types].head())
# print(pokemon.columns)

print(type(pokemon[["Type 1", "Type 2"] + types].head()))
print(type(pokemon[[["Type 1", "Type 2"] + types][0]].head()))