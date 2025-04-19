from typing import List
import requests
import json
import os
from sagemaker_client import generate_response  
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
import bert_score
import textstat
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
import torch
import nltk
from nltk import sent_tokenize
nltk.download('punkt', quiet=True)


def translate_text(text: str, max_new_tokens: int = 550, top_p: float = 0.9, temperature: float = 0.6, age="20", gender="male", ispregnant=False) -> str:
    """
    Constructs the translation prompt and calls the SageMaker endpoint.
    """
    pregnancy = "Pregnant" if ispregnant else "Not pregnant"
    system_prompt = (
        f"Please translate the following FDA information into a personalized, bullet-point list "
        f"**Do not plan to exceed 450 tokens in your response.**\n\n"
        "Include the following details: Drug Name, Ingredients, Purpose and Usage, Dosage and Administration, "
        "Adverse Ingredients (use 'ask doctor or pharmacist' if needed), and Warnings/Adverse Reactions.\n\n"
        "Return ONLY bullet points separated by new lines"
    )
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    # print(age, gender, ispregnant)
    response = generate_response(prompt, max_new_tokens, top_p, temperature)
    # Handle list responses (e.g., [ { "generated_text": ... } ])
    if isinstance(response, list) and response:
        first = response[0]
        if isinstance(first, dict):
            if 'generated_text' in first:
                return first['generated_text']
            elif 'text' in first:
                return first['text']
        elif isinstance(first, str):
            return first
    # Handle dict responses
    if isinstance(response, dict):
        if 'generated_text' in response:
            return response['generated_text']
        elif 'text' in response:
            return response['text']
        return next(iter(response.values()))
    # Fallback: assume string
    return response


# def fda_translate(
#         # user_id: str = Query(..., description="User ID for the current user"),
#         # medication: str = Query(..., description="Medication name to query FDA API"),
#         # max_new_tokens: int = Query(256, description="Max tokens for translation"),
#         # top_p: float = Query(0.9, description="Top p for translation"),
#         # temperature: float = Query(0.6, description="Temperature for translation")):
#     # Query FDA API using openFDA's drug label endpoint.
#     fda_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medication}&limit=1"
#     # retrieve user from MongoDB
#     user_age = user.get("age", "20")
#     user_gender = user.get("gender", "male")
#     is_pregnant = user.get("pregnant", False)

#     try:
#         fda_response = requests.get(fda_url)
#         fda_response.raise_for_status()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch FDA data: {str(e)}")
    
#     data = fda_response.json()
#     if "results" not in data or len(data["results"]) == 0:
#         raise HTTPException(status_code=404, detail="No FDA data found for the provided medication")
    
#     result = data["results"][0]
    
#     def extract_field(key: str):
#         value = result.get(key, "")
#         if isinstance(value, list):
#             return " ".join(value)
#         return value

#     combined_text = (
#         f"Purpose: {extract_field('purpose')}\n"
#         f"Indication and Usage: {extract_field('indication_and_usage')}\n"
#         f"Active Ingredient: {extract_field('active_ingredient')}\n"
#         f"Do not use: {extract_field('do_not_use')}\n"
#         f"Warnings: {extract_field('warnings')}\n"
#         f"Instruction For Use: {extract_field('instruction_for_use')}\n"
#         f"Drug Interactions: {extract_field('drug_interactions')}"
#         f"Dosage: {extract_field('dosage_and_administration')}"
#         f"Pregnancy or Breastfeeding: {extract_field('pregnancy_or_breast_feeding')}"
#         f"Ask Doctor: {extract_field('ask_doctor')}"
#         f"Ask Doctor or Pharmacist: {extract_field('ask_doctor_or_pharmacist')}"
#     )
#     try:
#         translation_result = translate_text(combined_text, max_new_tokens, top_p, temperature, user_age, user_gender, is_pregnant)
#         return translation_result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate translation: {str(e)}")

def get_combined_text(medication: str) -> str:
    """
    Helper to fetch FDA data and build the combined_text string for translation.
    """
    fda_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medication}&limit=1"
    response = requests.get(fda_url)
    response.raise_for_status()
    data = response.json()
    if "results" not in data or not data["results"]:
        raise ValueError(f"No FDA data for medication: {medication}")
    result = data["results"][0]
    def extract_field(key: str):
        value = result.get(key, "")
        if isinstance(value, list):
            return " ".join(value)
        return value
    combined_text = (
        f"Purpose: {extract_field('purpose')}\n"
        f"Indication and Usage: {extract_field('indication_and_usage')}\n"
        f"Active Ingredient: {extract_field('active_ingredient')}\n"
        f"Do not use: {extract_field('do_not_use')}\n"
        f"Warnings: {extract_field('warnings')}\n"
        f"Instruction For Use: {extract_field('instruction_for_use')}\n"
        f"Drug Interactions: {extract_field('drug_interactions')}\n"
        f"Dosage: {extract_field('dosage_and_administration')}\n"
        f"Pregnancy or Breastfeeding: {extract_field('pregnancy_or_breast_feeding')}\n"
        f"Ask Doctor: {extract_field('ask_doctor')}\n"
        f"Ask Doctor or Pharmacist: {extract_field('ask_doctor_or_pharmacist')}"
    )
    return combined_text

def evaluate_model_samples(csv_path: str = "medicationsdataset.csv"):
    """
    Reads test profiles from CSV and evaluates model outputs on ROUGE, BERTScore, and readability.
    Returns a pandas DataFrame with results for each sample.
    """
    df_profiles = pd.read_csv(csv_path)
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    for _, row in df_profiles.iterrows():
        med = row['Medication']
        age = str(row['Age'])
        gender = row['Gender']
        ispreg = str(row['Pregnancy']).lower() in ["yes", "true", "pregnant"]
        source_text = get_combined_text(med)
        generated_text = translate_text(source_text, age=age, gender=gender, ispregnant=ispreg)
        rouge_scores = scorer.score(source_text, generated_text)
        P, R, F1 = bert_score.score([generated_text], [source_text], lang="en", rescale_with_baseline=True)
        fk_grade = textstat.flesch_kincaid_grade(generated_text)
        results.append({
            'Medication': med,
            'Age': age,
            'Pregnancy': row['Pregnancy'],
            'Gender': gender,
            'Generated_Text': generated_text,
            'Rouge1_Recall': rouge_scores['rouge1'].recall,
            'Rouge2_Recall': rouge_scores['rouge2'].recall,
            'RougeL_Recall': rouge_scores['rougeL'].recall,
            'BERT_Precision': P[0].item(),
            'BERT_Recall': R[0].item(),
            'BERT_F1': F1[0].item(),
            'FleschKincaid_Grade': fk_grade
        ,'FactCC_Consistency': factcc_score(source_text, generated_text)
        })
        print(results[-1])
    return pd.DataFrame(results)

factcc_model_name = "manueldeprada/FactCC"
factcc_tokenizer = BertTokenizer.from_pretrained(factcc_model_name)
factcc_model = BertForSequenceClassification.from_pretrained(factcc_model_name)
factcc_model.eval()

def factcc_score(source: str, summary: str) -> float:
    """
    Splits summary into sentences and checks each one against the source using FactCC.
    Returns the percentage of consistent sentences.
    """
    summary_sentences = sent_tokenize(summary)
    consistent = 0
    total = len(summary_sentences)

    for sentence in summary_sentences:
        inputs = factcc_tokenizer(
            sentence,
            source,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        with torch.no_grad():
            logits = factcc_model(**inputs).logits
            predicted_label = torch.argmax(logits, dim=1).item()
            label = factcc_model.config.id2label[predicted_label]
            if label == "CORRECT":
                consistent += 1
    print(total, consistent)

    return consistent / total if total > 0 else 0.0

if __name__ == "__main__":
    # print(factcc_score("Take this pill once daily", "Take this pill once daily. Do not take more than one pill a day."))
    # df_results = evaluate_model_samples()
    # df_results.to_csv("evaluation_results.csv", index=False)
    print(textstat.flesch_kincaid_grade("This is a test sentence."))
    print(textstat.flesch_kincaid_grade)
    # print(df_results)