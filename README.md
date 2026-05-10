# pathology-llm

# Pathology Report Translator

Fine‑tuned Llama 3.1 8B to convert complex pathology reports into clear, empathetic patient explanations (8th grade reading level).

## 🔧 How It Works

- **Base model**: `unsloth/llama-3-8b-Instruct-bnb-4bit`
- **Fine‑tuning method**: QLoRA (Low‑Rank Adaptation) on an AMD MI300X GPU
- **Dataset**: 20 real medical notes with human‑written simplifications
- **Training loss**: 0.053 (final)

## 📦 Model

The fine‑tuned LoRA adapter is available on Hugging Face:  
[oussama-elidrysy/pathology-llm](https://huggingface.co/oussama-elidrysy/pathology-llm)

## 🚀 Usage

Load the adapter on top of the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
model = PeftModel.from_pretrained(base, "oussama-elidrysy/pathology-llm")
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
