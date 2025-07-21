from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def preprocess_text(texts, tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, padding='post')

def generate_text(seed_text, next_words, maxlen, tokenizer, model):
    for _ in range(next_words):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=maxlen, padding='post')
        predictions = model.predict(tokenized_input, verbose=0)
        predicted_word_index = np.argmax(predictions[0])
        predicted_word = tokenizer.index_word.get(predicted_word_index, "<OOV>")
        seed_text += " " + predicted_word
    return seed_text
