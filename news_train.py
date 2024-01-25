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
dfhaberler = pd.read_csv('TrainFiles/haberler.csv', encoding='utf-8')
dfhaberler['metin'] = dfhaberler['metin'].fillna('')
label_encoder = LabelEncoder()
dfhaberler['Etiket'] = label_encoder.fit_transform(dfhaberler['kategori'])
num_classes = len(label_encoder.classes_)
print(dfhaberler)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dfhaberler['metin'])
X_train, X_test, y_train, y_test = train_test_split(X, dfhaberler['Etiket'], test_size=0.2, random_state=42, stratify=dfhaberler['Etiket'])
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
def analyze_save(model, vectorizer, haberler, dosya_adı):
    haberler_vec = vectorizer.transform(haberler)
    tahminler = model.predict(haberler_vec)
    tahmin_siniflar = tf.keras.backend.eval(tf.argmax(tahminler, axis=1))
    sonuçlar_df = pd.DataFrame({'Metin': haberler, 'Tahmin Sınıfı': tahmin_siniflar})
    sonuçlar_df['Tahmin Etiketi'] = label_encoder.inverse_transform(tahmin_siniflar)
    sonuçlar_df.to_csv(dosya_adı, index=False)
haber_test = [
    "Meclis bu hafta yoğun mesaiye hazırlanıyor. 2024 Yılı Merkezi Yönetim Bütçe Kanunu Teklifi, yarın Genel Kurul'da görüşülmeye başlanacak. 14 gün sürecek görüşmelerde Genel Kurul, cumartesi ve pazar günleri de dahil kesintisiz toplanacak.",
    "Zonguldak'ta heyelan nedeniyle tek katlı müstakil evin yıkılması sonucu göçük altında kalan anne ve oğlunun cansız bedenine ulaşıldı.",
    "Yargıtay 11. Hukuk Dairesi, kredi sözleşmesine kefil olarak imza atan okuma yazma bilmeyen kişinin, sözleşme içeriğini bilmediğinin kabul edilemeyeceğine ve borçtan sorumlu olacağına hükmetti.",
    "Tokat-Sivas karayolunda minibüs kamyona arkadan çarptı. Kazada minibüste bulunan 5 kişi hayatını kaybetti."
]
analyze_save(model, vectorizer, haber_test, dosya_adı='Analysis/news_analysis.csv')
model.save('Modeller/news_model')
