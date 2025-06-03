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

# 1) Rhyme Word Generation inference function(P-CoT3 example)
def rhyme_inference(text, model, tokenizer):
    rhyme_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in American English phonology, phonetics, and morphology. "
                "Rhyming words share the same ending sound. Your task is to guide the student in discovering rhyming words through discovery learning."
            )
        },
        {
            "role": "user",
            "content": "I'm ready to learn about rhyming words and how to identify them."
        },
        {
            "role": "assistant",
            "content": (
                "Great! Let's start by understanding what rhyming words are. They share the same ending sounds. "
                "Now let's try an example together."
            )
        },
        # Example 1
        {
            "role": "user",
            "content": "Word: 'information'\n\nCan you think of any words that rhyme with 'information'?"
        },
        {
            "role": "assistant",
            "content": "Start by identifying the ending sound of 'information' and think of similar sounding words."
        },
        {
            "role": "user",
            "content": "Here are some words that rhyme with 'information': isolation, operation, conversation, corporation, demonstration."
        },
        {
            "role": "assistant",
            "content": "That's correct! Great examples."
        },
        # Example 2
        {
            "role": "user",
            "content": "Word: 'available'\n\nCan you think of any words that rhyme with 'available'?"
        },
        {
            "role": "assistant",
            "content": "Try to identify the ending sound and match with similar ones."
        },
        {
            "role": "user",
            "content": "Here are some words that rhyme with 'available': explainable, restrainable, retainable, retrainable, sustainable."
        },
        {
            "role": "assistant",
            "content": "Excellent work!"
        },
        # Example 3
        {
            "role": "user",
            "content": "Word: 'transport'\n\nCan you think of any words that rhyme with 'transport'?"
        },
        {
            "role": "assistant",
            "content": "Identify the ending sound and find similar words."
        },
        {
            "role": "user",
            "content": "Here are some words that rhyme with 'transport': passport, escort, report, resort, retort."
        },
        {
            "role": "assistant",
            "content": "Perfect!"
        },
        # Final request
        {
            "role": "user",
            "content": (
                "Now it's your turn! Word: '{text}'\n\n"
                "Provide exactly 5 DIFFERENT rhyming words, separated by commas. "
                f"Start your answer with: 'Here are some words that rhyme with {text}:'"
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        rhyme_messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 2) Inference from dataset
def process_rhyme(data, model_id, model, tokenizer):
    results = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Inference with {model_id}"):
        word = row[0]
        ground_truth = eval(row[1])  # List of rhyming words
        predicted_rhymes = rhyme_inference(word, model, tokenizer)

        results.append({
            'word': word,
            'ground_truth_rhymes': ground_truth,
            'predicted_rhymes': predicted_rhymes
        })

    return pd.DataFrame(results)


# 3) Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="your_model_id_here", help="Hugging Face model ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token for model access")
    args = parser.parse_args()

    rhyme_files = sorted([str(x) for x in Path('../benchmark_dataset/rhyme_word_generation/').glob('*.csv')])
    rhyme1 = pd.read_csv(rhyme_files[0], header=None)  # common
    rhyme2 = pd.read_csv(rhyme_files[1], header=None)  # rare

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

    # 1) Common words
    df_common = process_rhyme(rhyme1, args.model_id, model, tokenizer)
    df_common.to_csv(f"rhyme_{safe_model_id}_common_3cot.csv", index=False)

    # 2) Rare words
    df_rare = process_rhyme(rhyme2, args.model_id, model, tokenizer)
    df_rare.to_csv(f"rhyme_{safe_model_id}_rare_3cot.csv", index=False)

    print("\n=== ALL DONE! ===")
