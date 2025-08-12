import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformer_block import TransformerBlock
from positional_encoding import PositionalEncoding

# ==== CUDA / GPU SETUP ====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Using GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f" GPU setup error: {e}")
else:
    print(" No GPU found. Running on CPU.")

# ==== PATHS ====
MODEL_PATH = os.path.join("saved_model", "transformer_model.h5")
TOKENIZER_PATH = os.path.join("saved_model", "tokenizer.pkl")
CONFIG_PATH = os.path.join("saved_model", "config.pkl")

# ==== LOAD TOKENIZER ====
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==== LOAD CONFIG ====
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)
maxlen = config["maxlen"]

# ==== LOAD MODEL ====
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'PositionalEncoding': PositionalEncoding
}
model = load_model(MODEL_PATH, custom_objects=custom_objects)
print(" Model loaded.")

# ==== SAMPLING FUNCTION ====
def sample_with_top_k_top_p(predictions, temperature=1.0, top_k=50, top_p=0.9, use_top_k=True, use_top_p=True):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    probabilities = exp_preds / np.sum(exp_preds)

    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]

    if use_top_k and top_k > 0:
        sorted_indices = sorted_indices[:top_k]
        sorted_probs = sorted_probs[:top_k]

    if use_top_p:
        cumulative_probs = np.cumsum(sorted_probs)
        top_p_mask = cumulative_probs <= top_p
        if not np.any(top_p_mask):
            top_p_mask[0] = True  # ensure at least one token
        top_p_indices = sorted_indices[top_p_mask]
        top_p_probs = sorted_probs[top_p_mask]
    else:
        top_p_indices = sorted_indices
        top_p_probs = sorted_probs

    top_p_probs /= np.sum(top_p_probs)
    return np.random.choice(top_p_indices, p=top_p_probs)

# ==== TEXT GENERATION ====
def generate_text(seed_text, next_words, tokenizer, model, maxlen,
                  temperature=1.0, top_k=50, top_p=0.9, use_top_k=True, use_top_p=True):
    generated_text = seed_text

    for _ in range(next_words):
        tokenized_input = tokenizer.texts_to_sequences([generated_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=maxlen, padding='post')

        predictions = model.predict(tokenized_input, verbose=0)[0]
        predicted_index = sample_with_top_k_top_p(predictions, temperature, top_k, top_p, use_top_k, use_top_p)
        predicted_word = tokenizer.index_word.get(predicted_index, "<OOV>")
        generated_text += " " + predicted_word

    return generated_text

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    seed = "The man who is working hard he is"
    output = generate_text(
        seed_text=seed,
        next_words=50,
        tokenizer=tokenizer,
        model=model,
        maxlen=maxlen,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        use_top_k=True,
        use_top_p=True
    )
    print("\n📘 Generated Text:\n", output)
