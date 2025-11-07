from typing import List, Dict
from bert_score import score
from sacrebleu.metrics import BLEU, CHRF
from scipy.stats import kendalltau
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from nltk.translate.ribes_score import corpus_ribes
from automaton import rearrange_characters

class JapaneseTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-char")

    def tokenize(self, text):
        ids = self.tokenizer.encode(text)[1:-1]
        return [self.tokenizer.decode([i]).strip("#") for i in ids]

def compute_bert_score(translated_sentences:List[str], original_sentences:List[str]):
    P, R, F1 = score(translated_sentences, [[i] for i in original_sentences], model_type = "tohoku-nlp/bert-base-japanese-char", lang = "jp", num_layers = 11)
    return sum(P.tolist()) / len(P.tolist()), sum(R.tolist()) / len(R.tolist()), sum(F1.tolist()) / len(F1.tolist())

def compute_bleu_score(translated_sentences:List[str], original_sentences:List[str]):
    bleu = BLEU(tokenize = "char")
    return bleu.corpus_score(translated_sentences, [original_sentences])

def compute_chrf_score(translated_sentences:List[str], original_sentences:List[str]):
    chrf = CHRF()
    return chrf.corpus_score(translated_sentences, [original_sentences])

def compute_rouge_score(translated_sentences:List[str], original_sentences:List[str]):
    score = {}
    score["rouge1"] = {"precision":0, "recall":0, "fmeasure":0}
    score["rouge2"] = {"precision":0, "recall":0, "fmeasure":0}
    score["rougeL"] = {"precision":0, "recall":0, "fmeasure":0}
    tokenizer = JapaneseTokenizer()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = False, tokenizer = tokenizer)
    for i, j in zip(translated_sentences, original_sentences):
        result = scorer.score(i, j)
        for k in result:
            score[k]["precision"] += result[k].precision
            score[k]["recall"] += result[k].recall
            score[k]["fmeasure"] += result[k].fmeasure
    for k in score:
        score[k]["precision"] = score[k]["precision"] / len(translated_sentences)
        score[k]["recall"] = score[k]["recall"] / len(translated_sentences)
        score[k]["fmeasure"] = score[k]["fmeasure"] / len(translated_sentences)
    return score

def compute_ribes_score(translated_sentences:List[str], original_sentences:List[str]):
    tokenizer = JapaneseTokenizer()
    references = []
    hypotheses = []
    for i, j in zip(translated_sentences, original_sentences):
        hypotheses.append(tokenizer.tokenize(i))
        references.append([tokenizer.tokenize(j)])
    return corpus_ribes(references, hypotheses)

def compute_kendalltau_score(original_marks:List[Dict], valid_marks:List[Dict]):
    tau_value = 0
    for i in range(len(valid_marks)):
        original_order = [c[1] for c in rearrange_characters(original_marks[i]["characters"])]
        translation_order = [c[1] for c in rearrange_characters(valid_marks[i]["characters"])]
        tau, p_value = kendalltau(original_order, translation_order)
        tau_value += tau
    return tau_value / len(valid_marks)

def compute_pmr_score(original_marks:List[Dict], valid_marks:List[Dict]):
    pmr_value = 0
    for i in range(len(valid_marks)):
        original_order = [c[1] for c in rearrange_characters(original_marks[i]["characters"])]
        translation_order = [c[1] for c in rearrange_characters(valid_marks[i]["characters"])]
        pmr_value += sum([m == n for m, n in zip(original_order, translation_order)]) / len(translation_order)
    return pmr_value / len(valid_marks)