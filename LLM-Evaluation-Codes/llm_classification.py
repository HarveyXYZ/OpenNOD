import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Placeholder: API keys for 5 LLMs (to be filled by real values)
API_CONFIG = {
    "gpt4o-mini": {"key": "YOUR_KEY"},
    "gpt3.5-turbo": {"key": "YOUR_KEY"},
    "llama3.1-70b": {"key": "YOUR_KEY"},
    "moonshot-v1-8k": {"key": "YOUR_KEY"},
    "ernie-lite-8k": {"key": "YOUR_KEY"},
}

LLM_NAMES = list(API_CONFIG.keys())
NUM_LLMs = len(LLM_NAMES)

# Load SBERT for encoding examples
sbert_model = SentenceTransformer("all-mpnet-base-v2")

def call_llm_api(model_name, prompt):
    headers = {
        "Content-Type": "application/json",
    }

    if model_name == "gpt3.5-turbo":
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {API_CONFIG[model_name]['key']}"
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }

    elif model_name == "gpt4o-mini":
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {API_CONFIG[model_name]['key']}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }

    elif model_name == "llama3.1-70b":
        url = API_CONFIG[model_name]["endpoint"]  # Assume endpoint supports chat completion
        headers["Authorization"] = f"Bearer {API_CONFIG[model_name]['key']}"
        payload = {
            "model": "llama3-70b-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }

    elif model_name == "moonshot-v1-8k":
        url = "https://api.moonshot.cn/v1/chat/completions"
        headers["Authorization"] = f"Bearer {API_CONFIG[model_name]['key']}"
        payload = {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }

    elif model_name == "ernie-lite-8k":
        access_token = API_CONFIG[model_name]["key"]
        url = f"https://aip.baidubce.com/rpc/2.0/wenxinworkshop/chat/completions?access_token={access_token}"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_output_tokens": 5,
        }

    else:
        raise NotImplementedError(f"API for model {model_name} is not implemented.")

    # Send request and extract label
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise HTTPError for bad responses
    content = response.json()

    if model_name == "ernie-lite-8k":
        result = content.get("result", "")
    else:
        result = content['choices'][0]['message']['content']

    try:
        return int(result.strip())
    except ValueError:
        raise ValueError(f"Model {model_name} returned non-integer output: '{result}'")

def get_most_similar_examples(input_text, candidate_texts, k=3):
    all_texts = [input_text] + candidate_texts
    embeddings = sbert_model.encode(all_texts, convert_to_tensor=True).cpu().numpy()
    
    query_vec = embeddings[0]
    candidate_vecs = embeddings[1:]
    
    tokenized = [doc.split() for doc in candidate_texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(input_text.split())
    
    top_indices = np.argsort(scores)[::-1][:k]
    return [candidate_texts[i] for i in top_indices]

def construct_prompt(input_text, prompt_type="ZS", examples=None):
    task_description = "Classify the sentiment of the given text into one of five levels (1 to 5)."
    format_instruction = "Return only the label number (1-5) as output."

    if prompt_type == "ZS":
        prompt = f"{task_description}\nText: {input_text}\n{format_instruction}"
    elif prompt_type == "FS":
        demonstrations = "\n\n".join([f"Example:\nText: {ex}\nLabel: [X]" for ex in examples])
        prompt = f"{task_description}\n{demonstrations}\n\nNow classify:\nText: {input_text}\n{format_instruction}"
    else:
        raise ValueError("prompt_type must be 'ZS' or 'FS'")

    return prompt

def classify_with_all_llms(samples, prompt_type="ZS", example_pool=None):
    """
    Classify all samples using all LLMs.
    
    samples: list of str, input texts
    prompt_type: "ZS" or "FS"
    example_pool: list of candidate example texts for FS prompting
    """
    n = len(samples)
    output_matrix = np.zeros((NUM_LLMs, n), dtype=int)

    for j, model in enumerate(LLM_NAMES):
        for i, text in enumerate(samples):
            if prompt_type == "FS":
                examples = get_most_similar_examples(text, example_pool, k=3)
                prompt = construct_prompt(text, "FS", examples)
            else:
                prompt = construct_prompt(text, "ZS")

            label = call_llm_api(model, prompt)
            output_matrix[j, i] = label

    return output_matrix

