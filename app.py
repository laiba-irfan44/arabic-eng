import streamlit as st
import torch
from transformers import AutoTokenizer
from model import Transformer, PositionalEncoding  # Ensure these are imported

torch.serialization.add_safe_globals([Transformer, PositionalEncoding])

# Load Model & Vocabulary
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
vocab_data = torch.load("vocab.pth", map_location=DEVICE)
src_vocab = vocab_data["src_vocab"]
tgt_vocab = vocab_data["tgt_vocab"]

# Load trained model directly
model = torch.load("transformer_trans.pth", map_location=DEVICE, weights_only=False)
model.eval()

# Load tokenizer (same as Colab)
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Helper functions
def arabic_tokenizer(text):
    return tokenizer.tokenize(text)

def numericalize(sentence, vocab):
    tokens = arabic_tokenizer(sentence)
    return [vocab[SOS_TOKEN]] + [vocab.get(token, vocab[PAD_TOKEN]) for token in tokens] + [vocab[EOS_TOKEN]]

def generate_output(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    src_indices = numericalize(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab[SOS_TOKEN]]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = torch.argmax(output[0, -1, :]).item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab[EOS_TOKEN]:
            break

    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    generated_tokens = [inv_tgt_vocab[idx] for idx in tgt_indices if idx not in (tgt_vocab[SOS_TOKEN], tgt_vocab[EOS_TOKEN])]
    return " ".join(generated_tokens)

# Streamlit UI
st.title("Arabic to English Translation")
st.write("Enter an Arabic sentence and get its English translation.")

user_input = st.text_input("Enter Arabic Sentence", "")

if st.button("Translate"):
    if user_input.strip():
        with st.spinner("Translating..."):
            translation = generate_output(model, user_input, src_vocab, tgt_vocab, DEVICE)
        st.success("**English Translation:**")
        st.write(translation)
    else:
        st.warning("Please enter an Arabic sentence.")
