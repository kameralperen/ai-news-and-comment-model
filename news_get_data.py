from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import time
import requests

# Ana sayfa URL'si
url = 'https://www.webtekno.com/'

# Headless tarayıcı başlat
browser = webdriver.Chrome()

# Web sayfasını aç
browser.get(url)

count = 0

# Daha fazla haber yüklemek için butona tıkla
while count <= 4:
    try:
        # "Daha Fazla" butonunu bul
        more_button = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'content-timeline_more_text'))
        )

        # Butona tıkla
        more_button.click()

        # Sayfanın tam yüklenmesini bekle (istediğiniz süreyi ayarlayabilirsiniz)
        time.sleep(3)
        count += 1

    except Exception as e:
        # Buton bulunamazsa veya başka bir hata olursa döngüden çık
        break

# Sayfanın kaynak kodunu al
page_source = browser.page_source

# Headless tarayıcıyı kapat
browser.quit()

# BeautifulSoup kullanarak sayfa kaynağını işle
soup = BeautifulSoup(page_source, 'html.parser')

# Haber başlıklarını ve linklerini bulma
haber_basliklari = soup.find_all('a', class_='content-timeline__link')

# Daha önce çekilen içerikleri saklamak için bir küme (set) oluştur
icerikler = set()

# Dosyayı 'w' (write) modunda aç
with open('ExtractedData/news_content.txt', 'w', encoding='utf-8') as dosya:
    for baslik in haber_basliklari:
        haber_linki = baslik['href']

        # Her bir haberin sayfasına gitme ve içerik çekme
        haber_response = requests.get(haber_linki)
        haber_soup = BeautifulSoup(haber_response.text, 'html.parser')

        # İçerik divini bulma
        icerik_div = haber_soup.find('div', class_=re.compile("content-body__description"))

        if icerik_div:
            icerik = icerik_div.get_text(strip=True)  # Boşlukları temizleme

            # Eğer içerik daha önce eklenmediyse ekle
            if icerik not in icerikler:
                icerikler.add(icerik)
                dosya.write(icerik + '\n')