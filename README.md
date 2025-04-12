# 🚗 Turkish Text Generator: Automotive Domain

This project is a **Turkish language text generator** focused on the automotive domain, powered by a **Bidirectional LSTM neural network** using **TensorFlow/Keras**. It uses a combination of curated base text and dynamically retrieved content from Wikipedia to learn and generate coherent Turkish sentences about cars, electric vehicles, hybrid technologies, and more.

---

## 📚 Features

- Retrieves and cleans Turkish content from Wikipedia
- Tokenizes and prepares the text for model training
- Builds a deep neural network with:
  - Embedding Layer
  - Bidirectional LSTM
  - Dropout regularization
- Implements advanced text generation with:
  - Top-k sampling
  - Temperature scaling
- Saves the model and tokenizer for future use

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts words into dense vectors
- **Bidirectional LSTM**: Captures context in both directions
- **Dropout**: Helps prevent overfitting
- **Dense Layers**: Final output layer for word prediction

---

## 🧾 Dataset

- The core dataset consists of a long descriptive passage about automobiles (in Turkish).
- Additional content can be dynamically pulled from Wikipedia using `wikipediaapi`.

Topics include:
- Automotive history
- Electric vehicles
- Hybrid technologies
- Internal combustion engines
- Vehicle types and fuel technologies
- Autonomous driving systems

---

## 🚀 Text Generation

You can generate new Turkish text based on a starting seed phrase. The generation function allows:

- `next_words`: Number of words to generate
- `temperature`: Controls randomness in predictions
- `top_k`: Limits word choices to top-k probable options

Example:
```python
generate_text("Elektrikli araçların", next_words=30, temperature=0.7, top_k=15)
