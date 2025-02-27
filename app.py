import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

# Load the trained model
with open("code_to_pseudo_model.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def convert_cpp_to_pseudocode(cpp_code):
    inputs = tokenizer(cpp_code, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model.generate(**inputs, max_length=128)
    pseudocode = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pseudocode

# Gradio Interface
demo = gr.Interface(
    fn=convert_cpp_to_pseudocode,
    inputs=gr.Textbox(label="Enter C++ Code"),
    outputs=gr.Textbox(label="Generated Pseudocode"),
    title="C++ to Pseudocode Converter",
    description="Enter your C++ code and get the corresponding pseudocode using AI."
)

if __name__ == "__main__":
    demo.launch()
