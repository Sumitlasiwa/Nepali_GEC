from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

base_model = "google/mt5-small"
lora_adapter = "../outputs/best_model"
# lora_adapter = r"C:\Users\Lenovo\Desktop\Nepali_GEC\nepali_gec\outputs\best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()
    model.to(device)
    
    return tokenizer, model


def predict(text, tokenizer, model, max_new_tokens=64, num_beams=4):
    input_text = f"वाक्य सुधार्नुहोस्: {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_new_tokens,
            num_beams=num_beams,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer, model = load_model()

    while True:
        user_input = input("\n वाक्य लेख्नुहोस्: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        test_sentence = "नगरपालिका कस्तो किसिमको पर्यटक ल्याउन सक्छे "
        corrected = predict(user_input, tokenizer, model, max_new_tokens=64, num_beams=4)
        print(f"Original: {user_input}")
        print(f"Corrected: {corrected}")
