import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
data = pd.read_csv('cervix.csv')
# Menghapus kolom Id, dikarenakan kurang relevan
# data.drop("unamed column",axis=1,inplace=True)
data.drop("Unnamed: 0",axis=1,inplace=True)

# memisahkan atribut dan label
X = data[['behavior_sexualRisk', 'behavior_eating', 'behavior_personalHygiene',
          'intention_aggregation', 'intention_commitment', 'attitude_consistency',
          'attitude_spontaneity', 'norm_significantPerson', 'norm_fulfillment',
          'perception_vulnerability', 'perception_severity', 'motivation_strength',
          'motivation_willingness', 'socialSupport_emotionality', 'socialSupport_appreciation',
          'socialSupport_instrumental', 'empowerment_knowledge', 'empowerment_abilities',
          'empowerment_desires']]

# Target
y = data['ca_cervix']
 
# Membagi dataset menjadi data latih & data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# membuat model Gaussian naive bayes
gnb_model = GaussianNB()
 
# Melatih model dengan menggunakan data latih
gnb_model = gnb_model.fit(X_train, y_train)


with open('gnb_model.pkl', 'wb') as file:
    pickle.dump(gnb_model, file)