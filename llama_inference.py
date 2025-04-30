import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "llama_finetuned"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device="cuda:0",
)
messages = [
    {"role": "system", "content": "You are a smart grid security management expert, skilled in understanding and predicting grid security conditions"},
    {"role": "user", "content": "Please determine the current overload condition and power consumption (kW) based on the following information.\nThe current weather temperature is 19.1913 Celsius, and humidity is 38.3004.\nGiven the following historical charging data time series:\nVoltage (past|current) = [236.9291, 239.9761, 224.4519, 230.5807] | 234.5784 V;\nCurrent (past|current) = [24.5292, 39.2458, 19.6493, 41.844] | 21.2455 A;\nPower Factor (past|current) = [0.913, 0.9672, 0.8311, 0.9675] | 0.8326;\nReactive Power (past|current) = [1.174, 3.4639, 0.8904, 1.0849] | 1.6016 kVAR;\nVoltage Fluctuation (past|current) = [3.8385, 4.0509, 2.8421, -0.286] | -2.6559%;\nElectricity Price (past|current) = [0.2478, 0.4932, 0.1003, 0.4814] | 0.1806;\nOverload history status = [0, 0, 0, 0] (0 means no, 1 means yes);\nTransformer fault status (past|current) = [0, 0, 0, 0] | 0 (0 means normal, 1 means faulty);\nPower consumption in the past 1 hour = [5.8117, 9.4181, 4.4103, 9.6484] kW.\nPlease note!\nYour task is to determine the current overload condition (0 means no, 1 means yes) and predict power consumption (kW) by analyzing the given information and using your common sense.\nIn your answer, just provide your determination and predicted value.\n### Answer:"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
