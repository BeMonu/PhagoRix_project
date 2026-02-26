import torch
import json
import joblib
from tech.file_runner import resource_path
from model.model import model

#Further cross-validation of model for better realism of answear
class_coeffs = {
    "Citrobacter": 0.20,
    "Cronobacter": 0.33,
    "Enterobacter": 0.50,
    "Escherichia coli": 0.62,
    "Klebsiella pneumoniae": 0.88,
    "Salmonella": 0.40,
    "Serratia marcescens": 0.75,
    "Shigella": 0.13,
    "Yersinia": 0.20
}

MIN_COEF = 0.05

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
dictionary = {aa: i+1 for i, aa in enumerate(amino_acids)}
vocab_size = len(amino_acids) + 1

def encode_seq(seq, vocab, max_len):
    encoded = [vocab.get(aa, 0) for aa in seq]
    if len(encoded) >= max_len:
        return encoded[:max_len]
    return encoded + [0] * (max_len - len(encoded)) 
    
config_path = resource_path("model\\config.json")
with open(config_path, "r") as conf:
    config = json.load(conf)

encoder_path = resource_path("model\\label_encoder.pkl")
label_encoder = joblib.load(encoder_path)

model_path = resource_path("model\\lstm_amino_model.pth")
predicter = model(config["vocab_size"], config["embed_dim"], config["hidden_dim"], config["num_classes"])
predicter.load_state_dict(torch.load(model_path, map_location="cpu"))
predicter.eval()

def predict_protein(sequence, topk):
    sequence = sequence.upper()

    encoded = encode_seq(sequence, dictionary, config["max_len"])
    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        probs = predicter(x)

    topk_vals, topk_idxs = torch.topk(probs, topk, dim=1)
    
    results = []
    for idx, val in zip(topk_idxs[0], topk_vals[0]):

        label = label_encoder.inverse_transform([idx.item()])[0]
        coeff = class_coeffs.get(label, MIN_COEF)  # safe fallback
        adjusted_conf = val.item() * coeff
        adjusted_conf = max(adjusted_conf, MIN_COEF)
        results.append((label, adjusted_conf))

    return results
