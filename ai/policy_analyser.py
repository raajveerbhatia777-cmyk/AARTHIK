"""
AI Monetary Policy Stance & Sentiment Analyser.

Parses Reserve Bank of India (RBI) Monetary Policy Committee (MPC) statements
and extracts Policy Stance Scores (Hawkish / Dovish / Neutral).
"""

import re
from typing import Dict, List, Optional, Union


class PolicyAnalyser:
    """
    NLP & Lexicon-based Policy Stance Analyser for RBI MPC Statements.
    Calculates Hawkish, Dovish, and Net Policy Stance Index (0-100 scale).
    """

    HAWKISH_KEYWORDS = [
        "inflation risk", "upside risk", "price stability", "sticky inflation",
        "rate hike", "tightening", "withdrawal of accommodation", "overheating",
        "elevated yield", "cost push pressure", "supply constraints"
    ]

    DOVISH_KEYWORDS = [
        "growth slowdown", "downside risk", "demand weakness", "disinflation",
        "rate cut", "easing", "accommodative", "liquidity support",
        "revive growth", "slack", "softening inflation"
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def analyze_statement(self, text: str) -> Dict[str, Union[float, str, List[str]]]:
        """
        Parses policy statement text using rule-based keyword lexicon scoring.
        
        Parameters
        ----------
        text : str
            Raw text of RBI Monetary Policy statement or governor speech.

        Returns
        -------
        Dict
            Stance metrics, keyword matches, and net stance score (0-100).
            (0 = Extremely Dovish, 50 = Neutral, 100 = Extremely Hawkish)
        """
        if not text or not text.strip():
            return self._neutral_response()

        text_lower = text.lower()
        
        # Keyword counting
        hawkish_matches = [kw for kw in self.HAWKISH_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        dovish_matches = [kw for kw in self.DOVISH_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]

        hawkish_count = len(hawkish_matches)
        dovish_count = len(dovish_matches)
        total_signals = hawkish_count + dovish_count

        if total_signals == 0:
            return self._neutral_response()

        # Calculate normalized score (0 to 100)
        net_score = 50.0 + ((hawkish_count - dovish_count) / total_signals) * 50.0
        net_score = round(max(0.0, min(100.0, net_score)), 1)

        # Classify stance
        if net_score >= 60.0:
            stance_label = "Hawkish"
        elif net_score <= 40.0:
            stance_label = "Dovish"
        else:
            stance_label = "Neutral"

        return {
            "net_stance_score": net_score,
            "stance_label": stance_label,
            "hawkish_signals_count": hawkish_count,
            "dovish_signals_count": dovish_count,
            "hawkish_keywords_found": hawkish_matches,
            "dovish_keywords_found": dovish_matches,
        }

    @staticmethod
    def _neutral_response() -> Dict[str, Union[float, str, List[str]]]:
        return {
            "net_stance_score": 50.0,
            "stance_label": "Neutral",
            "hawkish_signals_count": 0,
            "dovish_signals_count": 0,
            "hawkish_keywords_found": [],
            "dovish_keywords_found": [],
        }


if __name__ == "__main__":
    analyser = PolicyAnalyser()
    
    sample_rbi_statement = """
    The Monetary Policy Committee noted that inflation risk remains elevated due to food price volatility, 
    necessitating sustained focus on price stability. The committee reiterated its stance on withdrawal 
    of accommodation to align inflation with the 4 percent target, despite persistent supply constraints.
    """

    result = analyser.analyze_statement(sample_rbi_statement)
    print("=== RBI Policy Stance Analysis Output ===")
    print(f"Stance Label     : {result['stance_label']}")
    print(f"Net Stance Score : {result['net_stance_score']} / 100")
    print(f"Hawkish Keywords : {result['hawkish_keywords_found']}")
    print(f"Dovish Keywords  : {result['dovish_keywords_found']}")
