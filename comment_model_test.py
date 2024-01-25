import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Kaydedilen modeli yükle
loaded_model = tf.keras.models.load_model('Modeller/comment_model')

with open('ExtractedData/comments_contents.txt', 'r', encoding='utf-8') as dosya:
    webtekno_haberler = dosya.readlines()

# Veri setini yükleyip etiketleri sayısal hale getir
df = pd.read_csv('TrainFiles/yorumlar.csv', encoding='utf-16')
df['yorum'] = df['yorum'].fillna('')
label_encoder = LabelEncoder()
df['Etiket'] = label_encoder.fit_transform(df['sınıf'])
num_classes = len(label_encoder.classes_)

# TF-IDF vektörleştirmeyi uygula
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['yorum'])


# Analiz yap ve sonuçları kaydet
def analysis_save(model, vectorizer, yorumlar, dosya_adı):
    yorumlar_vec = vectorizer.transform(yorumlar)
    tahminler = model.predict(yorumlar_vec)
    tahmin_siniflar = tf.keras.backend.eval(tf.argmax(tahminler, axis=1))
    sonuçlar_df = pd.DataFrame({'Yorum': yorumlar, 'Tahmin Sınıfı': tahmin_siniflar})
    sonuçlar_df['Tahmin Etiketi'] = label_encoder.inverse_transform(tahmin_siniflar)
    sonuçlar_df.to_csv(dosya_adı, index=False)



analysis_save(loaded_model, vectorizer, webtekno_haberler, dosya_adı='ResultsData/comment_result.csv')