import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_builder import build_transformer_model

# ==== CONFIGURATION ====
DATASET_DIR = "dataset"
NUM_WORDS = 10000
MAXLEN = 8024
NUM_LAYERS = 12
EMBED_DIM = 256
NUM_HEADS = 16
FF_DIM = 1024
DROPOUT_RATE = 0.1
EPOCHS = 3
BATCH_SIZE = 2
MODEL_NAME = "transformer_model"
OUTPUT_DIR = "saved_model"

# ==== LOAD TEXT DATA FROM FOLDER ====
def load_texts_from_folder(folder):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

texts = load_texts_from_folder(DATASET_DIR)

# ==== TOKENIZATION ====
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post')

# ==== SEQUENCE PREPARATION ====
flattened_sequences = [token for seq in sequences for token in seq]
X_train, y_train = [], []
for i in range(1, len(flattened_sequences)):
    X_train.append(flattened_sequences[:i])
    y_train.append(flattened_sequences[i])
X_train = pad_sequences(X_train, maxlen=MAXLEN, padding='post')
y_train = np.array(y_train)

# ==== MODEL BUILDING ====
model = build_transformer_model(
    vocab_size=NUM_WORDS,
    maxlen=MAXLEN,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    rate=DROPOUT_RATE,
    num_layers=NUM_LAYERS,
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ==== TRAINING ====
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# ==== SAVE OUTPUTS ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save Keras model in .h5 (for conversion)
model_path_h5 = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.h5")
model.save(model_path_h5)

# Save tokenizer
with open(os.path.join(OUTPUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# Save config
config = {
    "num_words": NUM_WORDS,
    "maxlen": MAXLEN,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "ff_dim": FF_DIM,
    "dropout_rate": DROPOUT_RATE,
    "num_layers": NUM_LAYERS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
}
with open(os.path.join(OUTPUT_DIR, "config.pkl"), "wb") as f:
    pickle.dump(config, f)

print(f"Model and assets saved to '{OUTPUT_DIR}'")
