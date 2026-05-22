## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework
## Name : K S Vinay Suhirthan
## Reg. No.: 212224230305
### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
To develop a Named Entity Recognition (NER) web application using the dslim/bert-base-NER model from Hugging Face that identifies and highlights entities such as persons, organizations, locations, and dates in user-provided text through an interactive Gradio interface.

### DESIGN STEPS:

#### STEP 1:
Import Libraries: Import necessary modules — os, requests, json, gradio, and dotenv.
#### STEP 2:
Load API Key: Load the Hugging Face API key using environment variables.
#### STEP 3:
Set API Endpoint: Define the model endpoint (dslim/bert-base-NER).
#### STEP 4:
Define Request Function: Create get_completion() to send text to the Hugging Face API and receive model predictions.
#### STEP 5:
Implement merge_tokens() to combine subword tokens into complete entities and build ner() to handle input text and format model output for display.
#### STEP 6:
Gradio Interface: Design a gr.Interface() with text input, highlighted entity output, and example texts.
Launch App: Run the Gradio app to interactively identify named entities in text.
### PROGRAM:
```python
import os
import requests
import json
import gradio as gr
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  
hf_api_key = os.environ.get("HF_API_KEY")
API_URL = os.environ.get(
    "HF_API_NER_BASE",  
    "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
)
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data["parameters"] = parameters

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    
    try:
        return response.json()  
    except json.JSONDecodeError:
        print("Invalid response from Hugging Face API:")
        print(response.text)
        return []

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        # Skip any malformed tokens
        if not all(k in token for k in ("start", "end", "entity", "word")):
            continue
        if merged_tokens and token["entity"].startswith("I-") and \
           merged_tokens[-1]["entity"].endswith(token["entity"][2:]):
            last = merged_tokens[-1]
            last["word"] += token["word"].replace("##", "")
            last["end"] = token["end"]
            last["score"] = (last["score"] + token["score"]) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not output or not isinstance(output, list):
        return {"text": input_text, "entities": []}
    merged = merge_tokens(output)
    return {"text": input_text, "entities": merged}
gr.close_all()

```

### OUTPUT:
<img width="1165" height="811" alt="image" src="https://github.com/user-attachments/assets/60a8f4d4-dea6-4f91-872f-92e2e0ee8e7c" />


### RESULT:
Thus the program todesign and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation was executed successfully.
