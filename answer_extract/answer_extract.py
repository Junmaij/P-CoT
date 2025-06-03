##### Rhyme Word Generation answer extract #####
def extract_rhyme(text):
    if "assistant" in text.lower():
        text = text.split("assistant")[-1].strip()
    if ":" in text:
        text = text.split(":", 1)[-1].strip()
    words = []
    if "\n" in text:
        lines = text.split("\n")
        for line in lines:
            # Remove leading numbers or Roman numerals
            line = re.sub(r"^(?:\d+|[IVXLCDM]+)\.\s*", "", line)  # 숫자, 로마 숫자 제거 
            if "," in line:
                words.extend(line.split(","))
            else:
                words.append(line)
    else:
        if "," in text:
            words = text.split(",")
        else:
            words = [text]
    # Clean and normalize the words
    words = [w.strip().rstrip(".").lower() for w in words if w.strip()]
    # Remove duplicates while preserving order
    words = list(dict.fromkeys(words))
    
    return words

def rhyme_acc(df):
    final_cnt = 0 
    aa = 0
    for i in range(len(df['predicted_rhymes'])):
        try:
            cnt = 0
            aa += 1
            predict_text = df['predicted_rhymes'][i]
            predicted_words = extract_rhyme(predict_text)
            
            ground_truth = df['ground_truth_rhymes'][i].lower()
            
            for w in predicted_words:
                if w in ground_truth:
                    cnt += 1
           
            final_cnt += cnt / 5
        except Exception as e:
            
            continue
    if aa == 0:
        return 0
    acc = final_cnt / aa * 100
    return acc



##### G2P answer extract #####
def remove_stress(phoneme):
        if pd.isna(phoneme):
            return ''
        return re.sub(r'[ˈˌ\.ːˑ/]|//|\s+', '', phoneme)

def g2p_acc(df, task):
    # Clean ground truth phonemes
    answer = list(df['ground_truth_phoneme'].apply(remove_stress))
    
    # Clean predicted phonemes with unified preprocessing
    predict = list(
        df['predicted_phoneme']
        .apply(remove_stress)
        .apply(lambda x: x.split('assistant')[-1].strip())
    )

    # Count matches
    cnt = 0
    for i in range(len(predict)):
        if answer[i] in predict[i]:
            cnt += 1

    # Compute accuracy
    acc = round(cnt / len(predict) * 100, 2)
    return acc




##### Syllable Counting answer extract #####
def syllable_acc(df):
    cnt = 0
    for i in range(len(df)):
        pred = str(df['predicted_syllable_count'][i])
        label = str(df['ground_truth_syllable_count'][i])
        
        # Extract only the final answer (e.g., after "assistant") if present
        if "assistant" in pred.lower():
            pred = pred.split("assistant")[-1].strip()
        
        # Check if label is contained in prediction string
        if label in pred:
            cnt += 1

    acc = round(cnt / len(df) * 100, 3)
    return acc
