import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import random
import wikipediaapi
import re

def get_wikipedia_content(topic, lang='tr'):
    """Wikipedia'dan belirli bir konunun içeriğini çeker"""
    wiki_wiki = wikipediaapi.Wikipedia(
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="MyTextGenerator/1.0"
    )
    page = wiki_wiki.page(topic)
    if page.exists():
        text = page.text

        text = re.sub(r'==.*?==', '', text)  
        text = re.sub(r'\{\{.*?\}\}', '', text)  
        text = re.sub(r'\n+', '\n', text)  
        return text.strip()
    return ""

base_text = """Arabalar, modern yaşamın en temel ulaşım araçlarından biri olup, insanlık tarihinde büyük bir dönüşüme neden olmuştur. İlk otomobilin icadı 19. yüzyılın sonlarına dayansa da, endüstriyel üretime geçiş ve kitlesel kullanım, Henry Ford’un 1908 yılında ürettiği Model T ile hız kazanmıştır. Seri üretim sayesinde otomobiller daha ulaşılabilir hale gelmiş, böylece toplumların ekonomik ve sosyal yapıları üzerinde büyük etkiler yaratmıştır.  

Otomobillerin tarihi, 18. yüzyılda buhar gücüyle çalışan araçlarla başlamıştır. Ancak, bu araçlar ağır ve verimsiz olduğu için geniş çapta kullanılamamıştır. 19. yüzyılın sonlarında, içten yanmalı motorların geliştirilmesiyle otomobil endüstrisi ivme kazanmıştır. Karl Benz, 1885 yılında modern otomobilin temellerini atan Benz Patent-Motorwagen’i üretmiş ve bu araç, günümüz arabalarının atası kabul edilmiştir. 20. yüzyılda ise Henry Ford’un üretim hattı sistemini kullanarak otomobilleri daha ucuz hale getirmesiyle, milyonlarca insanın araba sahibi olması sağlanmıştır.  

İkinci Dünya Savaşı’ndan sonra otomobil endüstrisi büyük bir büyüme yaşadı. 1950’ler ve 1960’larda Amerikan otomobilleri, büyük ve güçlü motorlarla donatılmış lüks araçlara evrildi. Avrupa’da ise Volkswagen Beetle ve Mini gibi ekonomik modeller popüler hale geldi. 1970’lerde petrol krizinin patlak vermesi, otomobil üreticilerini yakıt verimliliği konusunda yenilik yapmaya itti. 1980’ler ve 1990’lar, Japon otomobil üreticilerinin küresel pazarda yükselişine sahne oldu. Toyota, Honda ve Nissan gibi markalar, kalite, dayanıklılık ve yakıt ekonomisi açısından büyük ilerlemeler kaydetti.  

Geleneksel içten yanmalı motorlu arabalar, benzin veya dizel yakıtı kullanarak çalışır. Motor, yakıtı silindirler içinde patlatarak mekanik enerji üretir. Bu enerji, krank mili aracılığıyla şanzımana aktarılır ve araç tekerleklerine güç sağlar. İçten yanmalı motorlar genellikle dört zamanlı (emme, sıkıştırma, yanma, egzoz) prensibe dayanır. Ancak, günümüzde elektrikli motorlar daha popüler hale gelmektedir. Elektrikli araçlar, bataryalar aracılığıyla enerjiyi depolar ve elektrik motorları sayesinde tekerlekleri döndürerek çalışır.  

Arabalar, kullanım amacına ve tasarımına bağlı olarak farklı türlerde üretilmektedir:  

- **Sedan**: Günlük kullanım için en yaygın modeldir. Genellikle dört kapılıdır ve geniş bir iç mekana sahiptir.  
- **Hatchback**: Küçük ve kompakt yapısıyla özellikle şehir içi kullanım için idealdir.  
- **SUV (Sport Utility Vehicle)**: Yüksek sürüş pozisyonu, geniş iç hacmi ve arazi kabiliyeti ile öne çıkar.  
- **Spor Arabalar**: Performans odaklı araçlardır. Genellikle aerodinamik tasarıma, güçlü motorlara ve yüksek hız kapasitesine sahiptirler.  
- **Elektrikli ve Hibrit Araçlar**: Fosil yakıt tüketimini azaltarak çevre dostu bir ulaşım seçeneği sunar. Tesla, Nissan Leaf ve BMW i serisi gibi markalar bu alanda öncü konumundadır.  

Günümüzde otomobiller farklı enerji kaynakları ile çalışmaktadır:  

1. **Benzinli Motorlar**: En yaygın kullanılan motor türüdür. Yüksek devir kapasitesine sahiptir ancak yakıt tüketimi ve karbon emisyonu yüksektir.  
2. **Dizel Motorlar**: Daha düşük yakıt tüketimi ve yüksek tork üretimiyle özellikle uzun mesafelerde avantaj sağlar. Ancak, çevresel etkileri nedeniyle dizel motorlu araçların bazı ülkelerde yasaklanması gündemdedir.  
3. **Hibrit Araçlar**: İçten yanmalı motor ile elektrik motorunun bir araya gelmesiyle çalışır. Özellikle Toyota Prius gibi modeller hibrit teknolojisinin öncüsüdür.  
4. **Tam Elektrikli Araçlar**: Fosil yakıt kullanmadan, yalnızca elektrikle çalışır. Şarj edilebilir bataryaları sayesinde çevre dostudur.  
5. **Hidrojen Yakıt Hücreli Araçlar**: Hidrojeni enerji kaynağı olarak kullanarak su buharı dışında hiçbir emisyon üretmezler. Toyota Mirai ve Hyundai Nexo gibi modeller bu teknolojiyi kullanmaktadır.  

Günümüzde otomobiller, gelişmiş güvenlik sistemleri ile donatılmıştır:  

- **ABS (Anti-lock Braking System)**: Frenlerin kilitlenmesini önleyerek sürücünün aracın kontrolünü kaybetmesini engeller.  
- **ESP (Electronic Stability Program)**: Aracın kaymasını önleyerek stabiliteyi korur.  
- **Hava Yastıkları**: Kaza anında sürücü ve yolcuların yaralanmasını en aza indirmek için kullanılır.  
- **Çarpışma Önleme Sistemleri**: Sensörler ve kameralar aracılığıyla sürücüyü olası çarpışmalara karşı uyarır ve bazı durumlarda otomatik fren yapar.  
- **Otonom Sürüş Teknolojileri**: Tesla Autopilot, Mercedes-Benz Drive Pilot gibi sistemler, aracın belirli durumlarda sürücüsüz hareket edebilmesini sağlar.  

Otomotiv sektörü, teknolojik ilerlemelerle sürekli olarak evrilmektedir. Elektrikli araçların yaygınlaşmasıyla birlikte, şarj altyapılarının geliştirilmesi büyük bir önem taşımaktadır. Batarya teknolojilerinin gelişmesi, menzil sorununu azaltarak elektrikli araçların daha yaygın hale gelmesini sağlayacaktır.  

Bunun yanı sıra, otonom sürüş teknolojileri de hızla ilerlemektedir. Google’ın Waymo projesi ve Tesla’nın Full Self-Driving (FSD) sistemleri, tamamen sürücüsüz araçların gelecekte yollarda olacağının sinyallerini vermektedir. Ayrıca, akıllı şehir projeleriyle bağlantılı araç teknolojileri sayesinde, trafik yönetimi daha verimli hale gelecektir.  

Hidrojen yakıt hücreleri ve katı hal bataryaları gibi yeni enerji çözümleri, otomotiv sektörünün karbon nötr bir geleceğe ulaşmasını sağlayacaktır. Toyota, Honda ve BMW gibi markalar bu teknolojilere yatırım yapmaktadır.  

Sonuç olarak, arabalar sadece bir ulaşım aracı olmanın ötesine geçerek, modern yaşamın vazgeçilmez bir parçası haline gelmiştir. Gelecekte daha çevreci, daha akıllı ve daha güvenli araçlarla, otomobil dünyasının büyük bir dönüşüm geçireceği açıktır."""  # Buraya orijinal uzun metninizi ekleyin

wikipedia_topics = [
    "Otomobil"

]

wikipedia_texts = [get_wikipedia_content(topic) for topic in wikipedia_topics]
combined_text = base_text + "\n" + "\n".join(wikipedia_texts)

tokenizer = Tokenizer(filters='', lower=True)
tokenizer.fit_on_texts([combined_text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in combined_text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=256, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

def generate_text(seed_text, next_words=50, temperature=0.7, top_k=10):
    """Gelişmiş metin üretme fonksiyonu"""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]

        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        top_k_idx = np.argpartition(predictions, -top_k)[-top_k:]
        top_k_probs = predictions[top_k_idx]
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        predicted = np.random.choice(top_k_idx, p=top_k_probs)

        output_word = tokenizer.index_word.get(predicted, "")
        seed_text += " " + output_word

    seed_text = ' '.join(seed_text.split())  
    return seed_text.capitalize()

seed_phrases = [
    "Modern arabalarda",
    "Elektrikli araçların",
    "Otonom sürüş teknolojisi",
    "Hibrit motorların avantajları",
    "Gelecekte otomobiller"
]

for phrase in seed_phrases:
    generated = generate_text(
        seed_text=phrase,
        next_words=30,
        temperature=0.7,
        top_k=15
    )
    print(f"\nBaşlangıç: '{phrase}'\nÜretilen: {generated}\n{'-'*50}")

model.save("text_generator_model.h5")
with open('tokenizer.pkl', 'wb') as f:
    import pickle
    pickle.dump(tokenizer, f)