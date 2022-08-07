import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# scikit-learn version check
print(sklearn.__version__)

# scikit-learn is library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data pre-processing/
# model selection, model evaluation, and many other utilites

# Estimators : Also known as models or algorithm, estimators can be fitted to some data using its fit method

# Example of Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3], [11, 12, 13]]    # 2 samples, 3 features
y = [0, 1]                          # classes of each sample
clf.fit(X, y)
RandomForestClassifier(random_state=0)

# X is samples matirx(design matrix), I think if we use numpy or pandas, we make list(data) to ndarray or DataFrame
# Sample matrix have samples and features. In matrix (row, col) row mean how many samples are and column mean how many features are in data.
# y is target value as known as label. In regression y is real numbers for regression tasks, in classification y is integers(or discrete set of values)
# Above both y values are about supervised but in unsupervised y does't need to specified.(no labels(=target))
# y is usually 1d array where i the entry correspond to target of i in sample(row) of X(y의 index i는 X의 row index와 서로 대응된다)
# So, both X and y are usually numpy array or array-like dtypes, though some estimators(model) work with other format such as sparse matirces.(보통 X,y는 ndarray나 array 형태지만 모델에 따라 아닌 경우도 있다.)
# After estimators is fitted, can predict target from new data. Don't need retrain.
print(clf.predict(X))
print(clf.predict([[4, 5, 6], [14, 15, 16]]))

# Transformers and pre-processors
# Pipeline = pre-processing step(transform + impute) + ... + predict target
# In scikit-learn, pre-processor and transformers follow same API as estimator objects(inherit from the same BaseEstimator class)
# Transformer object don't have predict but transform method that make X to newly transformed sampel matrix X(트랜스포머 객체는 예측 기능은 없지만 샘플 행렬 X를 변환시키는 함수 transform(X) 을 가지고 있다.)
# From Youtube Liz Sander - Software Library APIs: Lessons Learned from scikit-learn - PyCon 2018 (https://www.youtube.com/watch?v=WCEXYvv-T5Q)
# In Youtube ETL(Extract(추출) + Transform(변환) + Load(적재 = 불러오기)) = Categorical Expansion + Null Imputation + Feature Scaling
# In Youtube Categorical Expansion + Null Imputation + Feature Scaling = In scikit-learn use class Transformers to perform those 3 steps

from sklearn.preprocessing import StandardScaler
X = [[0, 15],[1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)

# In Youtube ETL -> Train -> Validate -> Predict -> Money
# (Train -> Validate -> Predict) is subsumed into one class(Estimator) : maybe same as encapsulation
# Data -> Transform -> Model("Estimator") -> Money
# Estimators have two important methods : fit, predict
# Meta-estimator = Transformers + Moeld(Estimator)
# Meta-estimator - can grid search(fit many different options by parameters + give predict that best one)
# sk-learn API need encapsulation of the core pieces of supervised model
# Encapsulation : bundling of data with the methods that operate on that data, restricting of direct access to some of an object's components
# API is powerful about concepts of pipeline : put all of ETL estimator, meta estimator of whatever in single pipeline and treate as model itself
# pipeline can call fit and predict, works like any other model
# (API의 강력한 점은 pipeline이라는 개념인데, 이건 ETL 이든 meta 이든 뭐든 다 한 pipeline에 넣고 model 그 자체 처럼 사용할수 있다.
# pipeline은 fit, predict 함수를 부를수 있고, 다른 모델들처럼 쓸 수 있다.
# this pipeline abstract many compliacation(act of doing difficult) of different individual models that working different ways under the hoold(encapsulation)

# Sampel code loading iris dataset, spliting into train and test, computing accuracy score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1
# make estimator list and put it in (in Youtube)
# Pipeline is one of class 
est_list = [('scaler', StandardScaler()),('logistic', LogisticRegression())]
pipe = Pipeline(est_list)

# 2
# create a pipeline object
#pipe = make_pipeline(StandardScaler(), LogisticRegression())

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# fit the whole pipeline
print(pipe.fit(X_train, y_train))
# we can now use it like any other estimator
print(accuracy_score(pipe.predict(X_test), y_test))

# As you can see on above code, make_pipeline have 2 parameters that first one is Transformers and second one is estimators(predictors)
# Both can combined in single unifying object = Pipeline
# Pipeline also offer same API as regular estimator(like just regression) : fit, predict
# Also pipeline can prevent data leakage

# in Youtube she make method that make estimator parameter to variable

from sklearn.ensemble import GradientBoostingClassifier

def score_iris(est):
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X,y)
    
    est_list = [('scaler', StandardScaler()), ('your_estimator', est)]
    pipe = Pipeline(est_list)
    pipe.fit(train_X, train_y)
    scores = pipe.predict(test_X)
    return pipe, scores

gbt = GradientBoostingClassifier(n_estimators = 50)
pipe, scores = score_iris(gbt)
print(pipe)
print(scores)

# In youtube, also pipeline can in the pipeline
# First pipline that do some null imputation but score iris function don't do that things
# so it does imputation before gradient boosting classifier 

# Imputer는 0.23 버전에서 삭제되었으므로 이후에 사용하려면 sklearn.impute 를 사용해야한다.
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer

pipe_est = Pipeline([('imputer', Imputer()), ('gbt', GradientBoostingClassifier())])
pipe, scores = score_iris(pipe_est)
print("Pipelin in pipeline")
print(pipe)
print(scores)

# so pipeline just do task step by step, so we don't need to learn each things deeply to make complications

# Model evaluation
# only just use same test and training set can lead to cross-validation
# which mean it only predict test data(that already split one time) not new data
# so we make different test data each time to make model not overfitting


# From Aiffel
print("\n이제부터 Aiffel 내용")

import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2 * x - 3 * r.rand(100)
plt.scatter(x,y)
plt.show()

print(x.shape)
print(y.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)
# model.fit(x, y)
# it makes error

X = x.reshape(100,1)
# make x Matrix
model.fit(X,y)

x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100,1)
y_new = model.predict(X_new)
# Tip
X_ = x_new.reshape(-1,1)
print(X_.shape)

# 성능 평가
from sklearn.metrics import mean_squared_error

# MSE (평균 제곱 오차) : 오차의 제곱에 평균을 씌운것
# 분산과의 차이는 bias에 따라 달라지는데 bias = 0 이면 둘이 같다
# bias는 편항 우리 퍼셉트론에서 배웠던 b (-로 넘겨주는) 를 생각하면 된다.
# youtube 참고 : https://www.youtube.com/watch?v=pJCcGK5omhE&t=98s
error = mean_squared_error(y,y_new)
print(error)

plt.scatter(x, y, label='input data')
plt.plot(X_new, y_new, color='red', label='regression line')
plt.show()

# Toy dataset 사용해보기

from sklearn.datasets import load_wine
data = load_wine()
type(data)

# what is the bunch? - simillar with dic
print('data =', data)
print('data.keys() =', data.keys())
# label 
print('data.data =', data.data)
# attribue array
print('data.data.shape =', data.data.shape)
# shape
print('data.data.ndim =',data.data.ndim)
# dimension
print('data.target =', data.target)
# target = label
print('data.target.shape =', data.target.shape )
# shape of target
print('data.feature_names =', data.feature_names)
# feature names
print('len(data.feature_names) =', len(data.feature_names))
# feature 개수
print('data.target_names =',data.target_names)
# target(label) 이름
print('len(data.feature_names) =',len(data.feature_names))
# target의 개수
# print('data.DESC')
# print(data.DESCR)

# pandas 사용
import pandas as pd

pd.DataFrame(data.data, columns=data.feature_names)

X = data.data
y = data.target

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#타겟 벡터 즉 라벨인 변수명 y와 예측값 y_pred을 각각 인자로 넣습니다. 
print(classification_report(y, y_pred))
#정확도를 출력합니다. 
print("accuracy = ", accuracy_score(y, y_pred))

# wind data 사용해보기
from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape)
print(data.target.shape)

X_train = data.data[:142]
X_test = data.data[142:]
print(X_train.shape, X_test.shape)

y_train = data.target[:142]
y_test = data.target[142:]
print(y_train.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print("정답률=", accuracy_score(y_test, y_pred))

from sklearn.model_selection import train_test_split

result = train_test_split(X, y, test_size=0.2, random_state=42)

print(type(result))
print(len(result))

result[0].shape
result[1].shape
result[2].shape
result[3].shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)