import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Kaydedilen modeli yükle
loaded_model = tf.keras.models.load_model('Modeller/news_model')

with open('ExtractedData/news_content.txt', 'r', encoding='utf-8') as dosya:
    webtekno_haberler = dosya.readlines()

# Veri setini yükleyip etiketleri sayısal hale getir
df = pd.read_csv('TrainFiles/haberler.csv', encoding='utf-8')
df['metin'] = df['metin'].fillna('')
label_encoder = LabelEncoder()
df['Etiket'] = label_encoder.fit_transform(df['kategori'])
num_classes = len(label_encoder.classes_)

# TF-IDF vektörleştirmeyi uygula
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['metin'])


# Analiz yap ve sonuçları kaydet
def analyze_save(model, vectorizer, haberler, dosya_adı):
    haberler_vec = vectorizer.transform(haberler)
    tahminler = model.predict(haberler_vec)
    tahmin_siniflar = tf.keras.backend.eval(tf.argmax(tahminler, axis=1))
    sonuçlar_df = pd.DataFrame({'Metin': haberler, 'Tahmin Sınıfı': tahmin_siniflar})
    sonuçlar_df['Tahmin Etiketi'] = label_encoder.inverse_transform(tahmin_siniflar)
    sonuçlar_df.to_csv(dosya_adı, index=False)



analyze_save(loaded_model, vectorizer, webtekno_haberler, dosya_adı='ResultsData/news_result.csv')