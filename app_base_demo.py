import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading base Llama 3.1 8B...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
print("Model ready.")

def translate(report):
    prompt = f"""### Instruction:
Convert this medical note into a clear, empathetic explanation for a patient (8th grade reading level). Be honest but hopeful. Never add information not in the report.

### Report:
{report}

### Patient Explanation:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Patient Explanation:")[-1].strip()

with gr.Blocks(title="Pathology Report Translator") as demo:
    gr.Markdown("# 🩺 Pathology Report Translator\n**Base Llama 3.1 model – fine‑tuned version available on Hugging Face**")
    with gr.Row():
        report_input = gr.Textbox(label="Pathology Report", lines=15, placeholder="Paste the full report here...")
        output = gr.Textbox(label="Patient Explanation", lines=15, interactive=False)
    submit = gr.Button("Translate", variant="primary")
    submit.click(fn=translate, inputs=report_input, outputs=output)

demo.launch(share=True, server_port=7860)