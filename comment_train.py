import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


tf.compat.v1.disable_eager_execution()


dfyorumlar = pd.read_csv('TrainFiles/yorumlar.csv', encoding='utf-16')

dfyorumlar['yorum'] = dfyorumlar['yorum'].fillna('')

label_encoder = LabelEncoder()
dfyorumlar['Etiket'] = label_encoder.fit_transform(dfyorumlar['sınıf'])
num_classes = len(label_encoder.classes_)
print(dfyorumlar)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dfyorumlar['yorum'])

X_train, X_test, y_train, y_test = train_test_split(X, dfyorumlar['Etiket'], test_size=0.2, random_state=42, stratify=dfyorumlar['Etiket'])
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_test, y_test_one_hot))

y_pred_prob = model.predict(X_test)
y_pred = tf.keras.backend.eval(tf.argmax(y_pred_prob, axis=1))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karmaşıklık Matrisi')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(y_test.value_counts(), labels=label_encoder.classes_, autopct='%1.1f%%', startangle=140)
plt.title('Sınıf Dağılımı')
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
def analiz_ve_kaydet(model, vectorizer, yorumlar, dosya_adı):
    yorumlar_vec = vectorizer.transform(yorumlar)
    tahminler = model.predict(yorumlar_vec)
    tahmin_siniflar = tf.keras.backend.eval(tf.argmax(tahminler, axis=1))
    sonuçlar_df = pd.DataFrame({'Yorum': yorumlar, 'Tahmin Sınıfı': tahmin_siniflar})
    sonuçlar_df['Tahmin Etiketi'] = label_encoder.inverse_transform(tahmin_siniflar)
    sonuçlar_df.to_csv(dosya_adı, index=False)
testyorumlar = [
    "Film çok güzeldi, kesinlikle izlemenizi tavsiye ederim.",
    "Yılmaz Erdoğan ın en iyi filmi olmadığı kesin. hatta bana, film yapmak yerine şiir okuması gerektiğini düşündüren bi eser olmuş. fazla dramatik.",
    "Kesinlikle kötü bir film. Boşa vakit harcamayın.",
]
analiz_ve_kaydet(model, vectorizer, testyorumlar, dosya_adı='Analysis/comment_analysis.csv')
model.save('Modeller/comment_model')
