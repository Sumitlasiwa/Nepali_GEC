"""
Evaluation metrics for Nepali GEC
"""

import evaluate
import numpy as np
from collections import Counter
import re

def tokenize_nepali(text):
    """Tokenizes Nepali text: splits on spaces and removes punctuation."""
    # Remove punctuation commonly used in Nepali
    text = re.sub(r"[।,!?]", "", text)
    return text.strip().split()

def gleu_sentence(reference, prediction, max_n=4):
    """
    Compute sentence-level GEC-GLEU.
    Returns a score between 0 and 1.
    """
    ref_tokens = tokenize_nepali(reference)
    hyp_tokens = tokenize_nepali(prediction)
    
    # Adjust max_n for short sentences
    max_n = min(max_n, len(ref_tokens), len(hyp_tokens))
    if max_n == 0:
        return 0.0  # empty sentence
    
    scores = []
    for n in range(1, max_n+1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        precision = overlap / max(1, sum(hyp_ngrams.values()))
        recall = overlap / max(1, sum(ref_ngrams.values()))
        scores.append(min(precision, recall))
    return sum(scores) / max_n

def corpus_gec_gleu(references, predictions):
    """
    Compute corpus-level GEC-GLEU.
    `references` can be a list of strings or a list of single-item lists.
    """
    # Flatten single-reference lists
    refs_flat = [r[0] if isinstance(r, list) else r for r in references]
    
    scores = [gleu_sentence(r, p) for r, p in zip(refs_flat, predictions)]
    return float(np.mean(scores))

# Load metrics once
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
bertscore_metric = evaluate.load("bertscore")


def create_compute_metrics(tokenizer):
    """
    Factory function to create compute_metrics with tokenizer
    
    Usage:
        compute_metrics = create_compute_metrics(tokenizer)
    """
    
    def compute_metrics(eval_pred):
        """
        Compute BLEU, chrF, Correction Accuracy, and BERTScore for Nepali GEC.
        Handles both token IDs and plain text predictions.
        """
        predictions, labels = eval_pred

        # --- Handle tuple outputs (e.g., logits + labels) ---
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # --- If preds/labels are lists of strings, skip decoding ---
        if isinstance(predictions[0], str) and isinstance(labels[0], str):
            preds_clean = [p.strip() for p in predictions]
            refs_clean = [r.strip() for r in labels]
        else:
            # Convert to numpy arrays
            predictions = np.array(predictions)
            labels = np.array(labels)

            # Handle logits (vocab dimension)
            if predictions.ndim == 3:
                predictions = predictions.argmax(axis=-1)

            # Replace -100 with pad_token_id
            predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
            labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

            # Decode
            preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            preds_clean = [p.strip() for p in preds]
            refs_clean = [r.strip() for r in refs]

        
        metrics = {}

        # --- BLEU ---
        try:
            non_empty_indices = [i for i, (p, r) in enumerate(zip(preds_clean, refs_clean)) if p and r]
            if non_empty_indices:
                preds_bleu = [preds_clean[i] for i in non_empty_indices]
                refs_bleu = [[refs_clean[i]] for i in non_empty_indices]
                bleu_result = bleu_metric.compute(predictions=preds_bleu, references=refs_bleu)
                metrics["bleu"] = bleu_result["score"]
            else:
                metrics["bleu"] = 0.0
        except Exception as e:
            print(f"BLEU computation failed: {e}")
            metrics["bleu"] = 0.0

        # --- chrF ---
        try:
            chrf_result = chrf_metric.compute(predictions=preds_clean, references=refs_clean)
            metrics["chrf"] = chrf_result["score"]
        except Exception as e:
            print(f"chrF computation failed: {e}")
            metrics["chrf"] = 0.0

        # --- Correction Accuracy ---
        try:
            exact_matches = np.mean([p == r for p, r in zip(preds_clean, refs_clean)])
            metrics["correction_accuracy"] = exact_matches
        except Exception as e:
            print(f"Correction accuracy computation failed: {e}")
            metrics["correction_accuracy"] = 0.0

        # --- BERTScore ---
        try:
            non_empty_indices_bert = [i for i, (p, r) in enumerate(zip(preds_clean, refs_clean)) if p and r]
            if non_empty_indices_bert:
                preds_bert = [preds_clean[i] for i in non_empty_indices_bert]
                refs_bert = [refs_clean[i] for i in non_empty_indices_bert]
                bertscore_result = bertscore_metric.compute(
                    predictions=preds_bert,
                    references=refs_bert,
                    lang="ne",
                    model_type="microsoft/mdeberta-v3-base"
                )
                metrics["bertscore_f1"] = float(np.mean(bertscore_result["f1"]))
            else:
                metrics["bertscore_f1"] = 0.0
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            metrics["bertscore_f1"] = 0.0
            
        # --- GLEU (SacreBLEU) ---
        try:
            gleu_score = corpus_gec_gleu(refs_clean, preds_clean)
            metrics["gleu"] = gleu_score
        except Exception as e:
            print("GLEU failed:", e)
            metrics["gleu"] = 0.0

        # --- Print one sample for sanity ---
        if len(preds_clean) > 0:
            print(f"🔍 Sample - Pred: '{preds_clean[0][:50]}...' | Ref: '{refs_clean[0][:50]}...' | Match: {preds_clean[0] == refs_clean[0]}")

        return metrics
    
    return compute_metrics


if __name__ == "__main__":
    from config import Config
    from train import setup_model
    _, tokenizer = setup_model(Config)
    compute_metrics = create_compute_metrics(tokenizer)
    preds = ["मेरो नाम सन्तोष हो ।", "म स्कुल जान्छु ।", "म खाना खान्छु ।"]
    refs  = ["मेरो नाम सन्तोष हो ।", "म स्कुल जान्छु ।", "म खाना खान्छु ।"]
    print(compute_metrics((preds, refs)))