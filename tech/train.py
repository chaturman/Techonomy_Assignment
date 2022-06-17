import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, sklearn as sl
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

dataset  = pd.read_csv(r'~/Desktop/tech/data/data.csv')


dataset = dataset.rename(columns={'Score':'Credit_Score','Active':'Is_active_member','Products':'No_of_Products','Exited':'Churn','Card':'Has_credit_card'})

dataset.drop(labels=['Id'], axis=1, inplace=True)
dataset.drop(labels=['Row'], axis=1, inplace=True)
dataset.drop(labels=['Surname'], axis=1, inplace=True)  

# converting the categorical variable to dummy variables
dataset['Nationality'] = dataset['Nationality'].astype('category').cat.codes
dataset['Gender'] = dataset['Gender'].astype('category').cat.codes
dataset['Has_credit_card'] = dataset['Has_credit_card'].astype('category').cat.codes
dataset['Churn'] = dataset['Churn'].astype('category').cat.codes


target = 'Churn'
X = dataset.drop('Churn', axis = 1)
Y = dataset[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=123, stratify=Y)


forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, Y_train)

importances = forest.feature_importances_



features = dataset.drop(['Churn'], axis=1).columns
indices = np.argsort(importances)[::1]

feature_imp_df = pd.DataFrame({'Feature':features, 'Importance':importances})
five_top_features = ['Age','Salary','Credit_Score','Balance','No_of_Products']


clf = LogisticRegression(random_state = 0, solver = 'lbfgs').fit(X_train[five_top_features], Y_train )

clf.predict(X_test[five_top_features])
clf.predict_proba(X_test[five_top_features])
clf.score(X_test[five_top_features], Y_test)


pickle_filename = '../churnmodel.pkl'
with open (pickle_filename, 'wb') as file:
    pickle.dump(clf, file)


