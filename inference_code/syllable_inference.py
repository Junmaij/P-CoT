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

# 1) Load JSONL data
def json_parsing(json_file_path):
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            txt = json.loads(line)
            data.append(txt)
    return data

# 2) Syllable counting inference function (P-CoT3 example)
def syllable_inference(text, model, tokenizer):
    syllable_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in American English phonology and phonetics. "
                "A syllable is a unit of pronunciation having exactly one vowel sound. "
                "Your goal is to assist the student in counting the total number of syllables in a given sentence using discovery learning methods."
            )
        },
        {
            "role": "user",
            "content": "I'm ready to learn about syllables and how to count them in a sentence."
        },
        {
            "role": "assistant",
            "content": (
                "Great! Let's start by understanding what a syllable is. Remember, a syllable has one vowel sound. "
                "Now let's try an example together."
            )
        },
        # Example 1
        {
            "role": "user",
            "content": (
                "Sentence: \"Grace has resigned herself to simply completing the upbringing of her teenage daughter.\"\n\n"
                "How many syllables do you think it has?"
            )
        },
        {
            "role": "assistant",
            "content": "Let's break it down together. Start by identifying the vowel sounds in each word."
        },
        {
            "role": "user",
            "content": (
                "Grace (1), has (1), resigned (2), herself (2), to (1), simply (2), completing (3), "
                "the (1), upbringing (3), of (1), her (1), teenage (2), daughter (2)"
            )
        },
        {
            "role": "assistant",
            "content": "Exactly! Now, let's sum them all up."
        },
        {
            "role": "user",
            "content": "22"
        },
        {
            "role": "assistant",
            "content": "That's correct! There are 22 syllables in that sentence."
        },
        # Example 2
        {
            "role": "user",
            "content": (
                "Sentence: \"This story is about a young girl's redemption in a small town.\"\n\n"
                "How many syllables do you think it has?"
            )
        },
        {
            "role": "assistant",
            "content": "Let's work through this one as well. Identify the vowel sounds in each word."
        },
        {
            "role": "user",
            "content": (
                "This (1), story (2), is (1), about (2), a (1), young (1), girl's (1), redemption (3), "
                "in (1), a (1), small (1), town (1)"
            )
        },
        {
            "role": "assistant",
            "content": "Good job! Now add them together."
        },
        {
            "role": "user",
            "content": "16"
        },
        {
            "role": "assistant",
            "content": "Correct again! It has 16 syllables."
        },
        # Example 3
        {
            "role": "user",
            "content": (
                "Sentence: \"The one thing that hasn't happened is a proposal.\"\n\n"
                "How many syllables do you think it has?"
            )
        },
        {
            "role": "assistant",
            "content": "Let's do the same process and find the vowel sounds in each word."
        },
        {
            "role": "user",
            "content": (
                "The (1), one (1), thing (1), that (1), hasn’t (2), happened (2), is (1), a (1), proposal (3)"
            )
        },
        {
            "role": "assistant",
            "content": "Perfect! Now, sum them up."
        },
        {
            "role": "user",
            "content": "13"
        },
        {
            "role": "assistant",
            "content": "Well done! The total is 13 syllables."
        },
        # Final Request
        {
            "role": "user",
            "content": (
                f"Now it's your turn! Count the TOTAL number of syllables in the given sentence:\n\n"
                f"Sentence: \"{text}\"\n\n"
                "Remember to identify vowel sounds and add them up. Do not include any breakdown or additional explanation."
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        syllable_messages, 
        add_generation_prompt=True,    
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3) Inference over a dataset
def process_syllable_count(data, model_id, model, tokenizer):
    results = []
    for item in tqdm(data, desc=f"Inference with {model_id}"):
        sentence = item['Sentence']
        ground_truth = item['syllable_count']
        predicted_count = syllable_inference(sentence, model, tokenizer)

        results.append({
            'sentence': sentence,
            'ground_truth_syllable_count': ground_truth,
            'predicted_syllable_count': predicted_count,
            'model_id': model_id
        })

    return pd.DataFrame(results)

# 4) Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="your_model_id_here", help="Hugging Face model ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (if required)")
    args = parser.parse_args()

    syllable = [str(x) for x in Path('../benchmark_dataset/syllable_counting/').glob('*.json')]
    syllable1 = json_parsing(syllable[0])

    print(f"\n===== Loading {args.model_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        token=args.token
    )
    model.eval()

    safe_model_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', args.model_id)

    # Run inference
    df_syllable = process_syllable_count(syllable1, args.model_id, model, tokenizer)

    out_file = f"syllable_{safe_model_id}_3cot.csv"
    df_syllable.to_csv(out_file, index=False)
    print(f"-> Saved to {out_file}")

    print("\n=== ALL DONE! ===")
