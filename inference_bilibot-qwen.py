from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("ZheYiShuHua/bilibot-qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("ZheYiShuHua/bilibot-qwen-1.5B")

# Define the prompt template
alpaca_prompt = """
你是一位B站老用户，请使用暴躁的语言风格，对以下问题给出简短、机智的回答：

### 用户:
{}

### 输出:
"""

# Define the EOS token
EOS_TOKEN = tokenizer.eos_token

def generate_output(input_text):
    # Format the input text using the prompt template
    formatted_input = alpaca_prompt.format(input_text)
    
    # Tokenize the input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    # Generate the summary
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    
    # Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after "### Response:")
    response_start = summary.find("### 输出:")
    if response_start != -1:
        summary = summary[response_start + len("### 输出:"):].strip()
    
    return summary

# Prepare the input
input_text = "请问孕妇打人算群殴吗？"

output_text = generate_output(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")