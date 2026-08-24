# 🎬 Hibrit Film Öneri Sistemi

MovieLens 100K veri seti üzerinde, içerik bazlı ve işbirlikçi filtreleme (collaborative filtering) yöntemlerini birleştiren hibrit bir film öneri sistemi.

## 📊 Veri Seti
- Kaynak: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research)
- 100,000 puanlama, 943 kullanıcı, 1,682 film
- Her film 19 farklı tür kategorisinden birine veya birkaçına ait

## 🔍 Yöntem

### 1. İçerik Bazlı Filtreleme (Content-Based)
Film türlerini **TF-IDF** ile vektörleştirip **cosine similarity** ile benzer filmleri buluyor. "Bu filmi beğendiysen, türce benzer filmler" mantığı.

### 2. İşbirlikçi Filtreleme (Collaborative Filtering)
Kullanıcı-film puan matrisini **mean-centering** ile normalize edip (puanlama önyargısını azaltmak için), item-based cosine similarity hesaplıyor. "Bu filmi beğenenler başka neyi beğenmiş" mantığı — türden bağımsız, tamamen izleyici davranışına dayalı.

### 3. Hibrit Birleştirme
İki yöntemin skorlarını **rank-based normalize** edip **max-birleştirme** stratejisiyle topluyor: bir film, herhangi bir yöntemde güçlü çıkıyorsa öneri listesine giriyor. Bu, hem türce benzer hem de izleyici kitlesince benzer filmleri aynı listede sunuyor.

**Örnek (Star Wars için öneriler):**
- Return of the Jedi, Empire Strikes Back — hem tür hem izleyici kitlesi güçlü
- Raiders of the Lost Ark, Starship Troopers — izleyici davranışı güçlü



### 4. AI Destekli Film Özetleri
MovieLens veri setinde film özeti (plot) bulunmuyor. Bu eksikliği kapatmak için, önerilen her film için **Gemini API** (gemini-3.5-flash-lite) ile anlık kısa özet üretiliyor. Bu özetler modelin eğitim verisine dayalıdır, gerçek zamanlı bir kaynaktan çekilmez — bu yüzden arayüzde bir uyarı notuyla belirtiliyor.




## 🚀 Kurulum ve Çalıştırma

\`\`\`bash
git clone https://github.com/egemenoral1-jpg/film-oneri-sistemi.git
cd film-oneri-sistemi
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
Gemini API özetleri için, [aistudio.google.com/apikey](https://aistudio.google.com/apikey) adresinden ücretsiz bir API key al, repo kökünde `.env` dosyası oluştur:
\`\`\`
GEMINI_API_KEY=senin-api-key-buraya
\`\`\`
\`\`\`

Modeli oluşturmak için:
\`\`\`bash
python src/recommender.py
\`\`\`

Web arayüzünü başlatmak için:
\`\`\`bash
streamlit run app.py
\`\`\`

## 📁 Proje Yapısı
\`\`\`
film-oneri-sistemi/
├── notebooks/            # Keşif ve deney notebook'ları
├── src/
│   └── recommender.py    # Üretime hazır hibrit öneri pipeline'ı
├── app.py                # Streamlit web arayüzü
├── requirements.txt
└── README.md
\`\`\`

## 🛠️ Kullanılan Teknolojiler
Python · pandas · scikit-learn (TF-IDF, cosine similarity) · Streamlit