#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pandasをインポート
import pandas as pd


# In[2]:


#testデータtrainデータの読み込み
train = pd.read_csv("/Users/hosokawayoshihisa/code/ml-competition-template/data/input/train.csv")
test = pd.read_csv("/Users/hosokawayoshihisa/code/ml-competition-template/data/input/test.csv")


# In[3]:


#テストデータの中身確認
test.head()


# In[4]:


#学習データの中身確認
train.head()


# In[5]:


#学習データを特徴量と目的関数に分ける
train_x = train.drop(["Survived"], axis=1)
train_y = train["Survived"]


# In[6]:


#train_xの中身確認
train_x.head()


# In[7]:


#train_yの中身確認
train_y.head()


# In[8]:


#テストデータは特徴量のみなので、そのまま
test_x = test.copy()


# In[9]:


#念の為中身確認
test_x.head()


# In[10]:


#PassengerIdは予測に寄与する変数ではなく、入れたままだとモデルが意味のある変数と勘違いする可能性があるのため削除する
train_x = train_x.drop(["PassengerId"], axis=1)
test_x = test_x.drop(["PassengerId"], axis=1)


# In[11]:


#念の為中身確認
train_x.head()


# In[12]:


#念の為中身確認
test_x.head()


# In[13]:


#Name, Ticket, Cabinも上手く使えば予測に有用そうであるが、やや煩雑な処理が必要そうなので一旦これらの変数の列を削除する
train_x = train_x.drop(["Name", "Ticket", "Cabin"], axis=1)
test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=1)


# In[14]:


#念の為中身確認
train_x.head()


# In[15]:


#GBDT（勾配ブースティング）を利用する。GBDTでは文字列を取り扱えないため、SexとEmbarkedにlabel encordingを適用する
from sklearn.preprocessing import LabelEncoder


# In[16]:


for c in ["Sex", "Embarked"]:
    #fit()を利用して学習データに基づいてどう変換するのか定める
    le = LabelEncoder()
    le.fit(train_x[c].fillna("NA"))
    
    #学習データ、テストデータを変換する
    train_x[c] = le.transform(train_x[c].fillna("NA"))
    test_x[c] = le.transform(test_x[c].fillna("NA"))


# In[17]:


#念の為中身確認
train_x.head()


# In[18]:


#念の為中身確認
test_x.head()


# In[19]:


#GBDTのライブラリのひとつであるXgboostを用いて、モデルを作成
from xgboost import XGBClassifier


# In[20]:


#モデルの作成及び学習データを与えての学習
model = XGBClassifier(n_estimators=20, random_state=71)


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


#テストデータの予測値を確率で出力する
pred = model.predict_proba(test_x)[:, 1]


# In[ ]:




