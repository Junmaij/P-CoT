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
from transformers import pipeline

# 에러문 일단 무시하도록 설정 (attention mask and the pad token id 어쩌구)
logging.getLogger("transformers").setLevel(logging.ERROR)


# 1) 데이터 로드 함수
def json_parsing(json_file_path):
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            txt = json.loads(line)
            data.append(txt)
    return data


#2) G2P 인퍼런스 함수
def gcp_inference_gemma(text, pipe):
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert in American English phonology, phonetics, and morphology. "
                "In this task, you are required to map a sequence of graphemes (characters representing a word) "
                "to a transcription of that word’s pronunciation (phonemes). "
                "If you encounter a word you cannot transcribe, respond with - NONE."
                "Consider the process of converting graphemes to phonemes. "
                "For instance, how would you transcribe the word 'apparently'? "
                "Please provide the phonemic representation."
            )
        },
        {
            "role": "assistant", 
            "content": "/əpˈɛɹəntli/"
        },
        # 최종 요청
        {
            "role": "user", 
            "content": (
                "Now, apply your understanding to this new word. "
                f"How would you transcribe the grapheme '{text}' into its phonemes? "
                "Remember, provide only the phoneme transcription of the word as a whole."
            )
        }
    ]
    
    try:
        # 파이프라인 호출
        outputs = pipe(messages, max_new_tokens=128)
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
        return assistant_response
    except Exception as e:
        print("Error during inference:", e)
        return None


#3) 데이터셋으로 인퍼런스 수행 함수 
def process_g2p(data, model_id, pipe):
    results = []
    for item in tqdm(data, desc=f"Inference with {model_id}"):
        grapheme = item['grapheme']
        ground_truth = item['outputs'].strip()  # 정답 phoneme
        predicted_phoneme = gcp_inference_gemma(grapheme, pipe)
        results.append({
            'grapheme': grapheme,
            'ground_truth_phoneme': ground_truth,
            'predicted_phoneme': predicted_phoneme,
            'model_id': model_id
        })
    return pd.DataFrame(results)


#5) 실행 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--token", type=str, default="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")
    args = parser.parse_args()

    gcp = [str(x) for x in Path('./llm_phonology/g2p/').glob('*.json')]
    gcp1 = json_parsing(gcp[0])             # low frequency 
    gcp2 = json_parsing(gcp[1])             # high frequency
    
    token = "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"

    print(f"\n===== Loading {args.model_id} =====")

    # 모델명 슬래시 같은거 제거 
    safe_model_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', args.model_id)

        pipe = pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        token=token
    )
    pipe.model.eval()

        # 1) gcp1 (저빈도) 인퍼런스
    #####################################
    df_gcp1 = process_g2p(gcp1, args.model_id, pipe)
    out_file1 = f"g2p_{safe_model_id}_low_1cot.csv"
    df_gcp1.to_csv(out_file1, index=False)
        

        #####################################
        # 2) gcp2 (고빈도) 인퍼런스
    #####################################
    df_gcp2 = process_g2p(gcp2, args.model_id, pipe)
    out_file2 = f"g2p_{safe_model_id}_high_1cot.csv"
    df_gcp2.to_csv(out_file2, index=False)

    
    print("\n=== ALL DONE! ===")