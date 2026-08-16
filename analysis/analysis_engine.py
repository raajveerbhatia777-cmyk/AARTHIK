import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

MACRO_KEYWORDS = {
    "hawkish": ["inflation", "rate hike", "tightening", "repo rate increase", "drain liquidity"],
    "dovish": ["growth", "rate cut", "accommodation", "support demand", "inject liquidity"],
    "cir_terms": ["repo rate", "reverse repo", "sdf", "msf", "laf", "corridor", "call money", "cir"]
}

def analyze(text: str) -> dict:
    doc = nlp(text)
    lower_text = text.lower()
    
    # 1. Sentiment Analysis via NLTK
    sentiment_scores = sia.polarity_scores(text)
    compound = sentiment_scores['compound']
    
    # 2. RBI Policy Stance Detection
    hawkish_hits = [w for w in MACRO_KEYWORDS["hawkish"] if w in lower_text]
    dovish_hits = [w for w in MACRO_KEYWORDS["dovish"] if w in lower_text]
    cir_hits = [w for w in MACRO_KEYWORDS["cir_terms"] if w in lower_text]
    
    if len(hawkish_hits) > len(dovish_hits):
        policy_stance = "Hawkish (Rate Hike / Tightening Risk)"
    elif len(dovish_hits) > len(hawkish_hits):
        policy_stance = "Dovish (Rate Cut / Accommodative)"
    else:
        policy_stance = "Neutral / Balanced"
        
    normalized_score = round((compound + 1) * 50)
    entities = list(set([f"{ent.text} ({ent.label_})" for ent in doc.ents]))
    keywords = list(set([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]))[:6]

    return {
        "score": normalized_score,
        "tone": policy_stance,
        "summary": f"Detected {len(cir_hits)} CIR policy terms. Stance classified as {policy_stance}.",
        "key_findings": [
            f"policy_stance:{policy_stance.split()[0].lower()}",
            f"cir_relevant:{len(cir_hits) > 0}"
        ],
        "details": {
            "stub": False,
            "cir_terms_found": cir_hits,
            "hawkish_signals": hawkish_hits,
            "dovish_signals": dovish_hits,
            "entities": entities,
            "keywords": keywords,
            "vader_scores": sentiment_scores
        }
    }
