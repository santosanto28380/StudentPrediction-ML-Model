import pandas as pd
data=pd.read_csv("Dataset/student.csv")
print(data)

import pandas as pd
data=pd.read_csv("Dataset/student.csv")
data=data.dropna(subset=['result'])
data=data.fillna(data.median(numeric_only=True))
print(data)
print(data.isnull().sum())

data.to_csv("Dataset/cleaned_student.csv",index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x=data.drop('result',axis=1)
y=data['result']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

accuracy=model.score(x_test,y_test)
print("Accuracy:",accuracy)
from sklearn.metrics import precision_score,recall_score, f1_score

y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred, pos_label='Pass')
recall = recall_score(y_test, y_pred, pos_label='Pass')
f1 = f1_score(y_test, y_pred, pos_label='Pass')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

import pickle
model.fit(x_train,y_train)
with open("models/student_model.pkl","wb") as file:
    pickle.dump(model,file)
    print("model saved successfully")
    with open("models/student_model.pkl","rb") as file:
        loaded_model=pickle.load(file)
        print("model loaded sucessfully")
