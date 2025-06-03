#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1) Load data from JSONL format
def json_parsing(json_file_path):
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            txt = json.loads(line)
            data.append(txt)
    return data


# 2) G2P conversion inference function (P-CoT3 example)
def gcp_inference(text, model, tokenizer):
    gcp_messages = [
        {
            "role": "system",
            "content": (
                "Act like an expert in American English phonology, phonetics, and morphology. "
                "Transcribe a sequence of graphemes (the characters representing a word) into their corresponding phonemes (the sounds that make up the pronunciation). "
                "If you encounter a word you cannot transcribe, respond with - NONE.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "Consider the process of converting graphemes to phonemes. "
                "For instance, how would you transcribe the word 'apparently'? "
                "Provide the phonemic representation."
            )
        },
        {
            "role": "assistant",
            "content": "/əpˈɛɹəntli/"
        },
        {
            "role": "user",
            "content": (
                "Let's try another example. "
                "How would you transcribe the word 'calorie'? "
                "Provide only the phoneme transcription."
            )
        },
        {
            "role": "assistant",
            "content": "/kˈælɚi/"
        },
        {
            "role": "user",
            "content": (
                "One more example for practice. "
                "Provide the phoneme transcription for the word 'freshman'."
            )
        },
        {
            "role": "assistant",
            "content": "/fɹˈɛʃmən/"
        },
        {
            "role": "user",
            "content": (
                "Now, apply your understanding to this new word. "
                f"How would you transcribe the grapheme '{text}' into its phonemes? "
                "Remember, provide only the phoneme transcription of the word as a whole."
            )
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        gcp_messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 3) Inference over a dataset
def process_g2p(data, model_id, model, tokenizer):
    results = []
    for item in tqdm(data, desc=f"Inference with {model_id}"):
        grapheme = item['grapheme']
        ground_truth = item['outputs'].strip()
        predicted_phoneme = gcp_inference(grapheme, model, tokenizer)

        results.append({
            'grapheme': grapheme,
            'ground_truth_phoneme': ground_truth,
            'predicted_phoneme': predicted_phoneme,
            'model_id': model_id
        })
    return pd.DataFrame(results)


# 4) Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="your_model_id_here", help="Hugging Face model ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (if required)")
    args = parser.parse_args()

    # Load datasets
    gcp_files = [str(x) for x in Path('../benchmark_dataset/g2p/').glob('*.json')]
    gcp1 = json_parsing(gcp_files[0])  # Low-frequency words
    gcp2 = json_parsing(gcp_files[1])  # High-frequency words

    print(f"\n===== Loading model: {args.model_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        token=args.token
    )
    model.eval()

    # Clean model name for output filename
    safe_model_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', args.model_id)

    # Run inference on low-frequency dataset
    df_gcp1 = process_g2p(gcp1, args.model_id, model, tokenizer)
    out_file1 = f"g2p_{safe_model_id}_low_3cot.csv"
    df_gcp1.to_csv(out_file1, index=False)

    # Run inference on high-frequency dataset
    df_gcp2 = process_g2p(gcp2, args.model_id, model, tokenizer)
    out_file2 = f"g2p_{safe_model_id}_high_3cot.csv"
    df_gcp2.to_csv(out_file2, index=False)

    print("\n=== ALL DONE! ===")
