# pip install sentence-transformers bert-score nltk
from functools import lru_cache

from transformers.utils import logging

logging.set_verbosity_error()

import nltk
from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# nltk.download('punkt_tab')
# nltk.download('punkt')


class Calculator:
    def __init__(self, w_bert=0.3, w_cos=0.5, w_kw=0.2):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scorer = BERTScorer(
            lang="en",
            model_type="roberta-large",
        )
        self.w_bert, self.w_cos, self.w_kw = w_bert, w_cos, w_kw
        
    def bert_f1(self, cand, gt):
        P, R, F1 = self.scorer.score([cand], [gt])
        return F1.item()
        
    def calculate(self, candidate_text, gt_text):
        if candidate_text is None or candidate_text.strip() == "":
            return 0.0
        
        bert_f1 = self.bert_f1(candidate_text, gt_text)
        
        cand_emb = self.model.encode(candidate_text, convert_to_tensor=True)
        gt_emb = self.model.encode(gt_text, convert_to_tensor=True)
        cos_sim = util.cos_sim(cand_emb, gt_emb).item()
        
        gt_tokens = word_tokenize(gt_text.lower())
        cand_tokens = word_tokenize(candidate_text.lower())
        matched = sum(1 for tok in gt_tokens if tok in cand_tokens)
        kw_coverage = matched / len(gt_tokens)
        
        score = self.w_bert * bert_f1 + self.w_cos * cos_sim + self.w_kw * kw_coverage
        
        return score

