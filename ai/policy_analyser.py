"""
AI Monetary Policy Stance & Sentiment Analyser with Negation & Weighting.

Parses Reserve Bank of India (RBI) Monetary Policy Committee (MPC) statements,
handles negation contexts, applies keyword importance weighting, and calculates
a standardized Policy Stance Index (0-100 scale).
"""

import re
from typing import Dict, List, Tuple, Union, Optional


class PolicyAnalyser:
    """
    NLP, Lexicon, and Rule-based Policy Stance Analyser for RBI MPC Statements.
    Calculates weighted Hawkish, Dovish, and Net Policy Stance Index (0-100 scale).
    """

    HAWKISH_KEYWORDS: Dict[str, float] = {
        "rate hike": 2.0,
        "tightening": 1.8,
        "withdrawal of accommodation": 2.0,
        "inflation risk": 1.5,
        "upside risk": 1.2,
        "price stability": 1.0,
        "sticky inflation": 1.5,
        "overheating": 1.5,
        "elevated yield": 1.2,
        "cost push pressure": 1.2,
        "supply constraints": 1.0,
    }

    DOVISH_KEYWORDS: Dict[str, float] = {
        "rate cut": 2.0,
        "easing": 1.8,
        "accommodative": 2.0,
        "growth slowdown": 1.5,
        "downside risk": 1.2,
        "demand weakness": 1.5,
        "disinflation": 1.5,
        "liquidity support": 1.5,
        "revive growth": 1.5,
        "slack": 1.2,
        "softening inflation": 1.2,
    }

    NEGATION_WORDS: set = {
        "no", "not", "never", "neither", "nor", "without", "lack", "hardly", "scarcely", "little"
    }

    def __init__(self, negation_window: int = 3, api_key: Optional[str] = None) -> None:
        self.negation_window = negation_window
        self.api_key = api_key

    def _is_negated(self, text: str, match_start_idx: int) -> bool:
        """Checks if a negation word exists within N words before the match index."""
        preceding_text = text[:match_start_idx].strip()
        words = re.findall(r"\b\w+\b", preceding_text.lower())
        window_words = words[-self.negation_window:] if len(words) >= self.negation_window else words
        return any(word in self.NEGATION_WORDS for word in window_words)

    def analyze_statement(self, text: str) -> Dict[str, Union[float, str, List[str]]]:
        """
        Parses policy statement text, handles negation flipping, and applies keyword weighting.

        Parameters
        ----------
        text : str
            Raw text of RBI Monetary Policy statement or governor speech.

        Returns
        -------
        Dict
            Weighted stance metrics, detected signals, and net stance score (0-100).
        """
        if not text or not text.strip():
            return self._neutral_response()

        text_lower = text.lower()
        
        weighted_hawkish = 0.0
        weighted_dovish = 0.0
        hawkish_signals: List[str] = []
        dovish_signals: List[str] = []

        # Process Hawkish Keywords
        for kw, weight in self.HAWKISH_KEYWORDS.items():
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
            for match in pattern.finditer(text_lower):
                if self._is_negated(text_lower, match.start()):
                    # Negated hawkish becomes dovish signal
                    weighted_dovish += weight
                    dovish_signals.append(f"NOT {kw} (flipped dovish)")
                else:
                    weighted_hawkish += weight
                    hawkish_signals.append(kw)

        # Process Dovish Keywords
        for kw, weight in self.DOVISH_KEYWORDS.items():
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
            for match in pattern.finditer(text_lower):
                if self._is_negated(text_lower, match.start()):
                    # Negated dovish becomes hawkish signal
                    weighted_hawkish += weight
                    hawkish_signals.append(f"NOT {kw} (flipped hawkish)")
                else:
                    weighted_dovish += weight
                    dovish_signals.append(kw)

        total_weight = weighted_hawkish + weighted_dovish

        if total_weight == 0.0:
            return self._neutral_response()

        # Calculate normalized score (0.0 = Dovish, 50.0 = Neutral, 100.0 = Hawkish)
        net_score = 50.0 + ((weighted_hawkish - weighted_dovish) / total_weight) * 50.0
        net_score = round(max(0.0, min(100.0, net_score)), 1)

        if net_score >= 60.0:
            stance_label = "Hawkish"
        elif net_score <= 40.0:
            stance_label = "Dovish"
        else:
            stance_label = "Neutral"

        return {
            "net_stance_score": net_score,
            "stance_label": stance_label,
            "weighted_hawkish_score": round(weighted_hawkish, 2),
            "weighted_dovish_score": round(weighted_dovish, 2),
            "hawkish_signals_found": hawkish_signals,
            "dovish_signals_found": dovish_signals,
        }

    @staticmethod
    def _neutral_response() -> Dict[str, Union[float, str, List[str]]]:
        return {
            "net_stance_score": 50.0,
            "stance_label": "Neutral",
            "weighted_hawkish_score": 0.0,
            "weighted_dovish_score": 0.0,
            "hawkish_signals_found": [],
            "dovish_signals_found": [],
        }


if __name__ == "__main__":
    analyser = PolicyAnalyser()

    sample_negated_statement = """
    The MPC noted that there is no inflation risk and no rate hike is required. 
    Instead, the committee sees significant growth slowdown requiring liquidity support.
    """

    result = analyser.analyze_statement(sample_negated_statement)
    print("=== Enhanced Policy Analyser Output ===")
    print(f"Stance Label       : {result['stance_label']}")
    print(f"Net Stance Score   : {result['net_stance_score']} / 100")
    print(f"Hawkish Weighted   : {result['weighted_hawkish_score']}")
    print(f"Dovish Weighted    : {result['weighted_dovish_score']}")
    print(f"Hawkish Signals    : {result['hawkish_signals_found']}")
    print(f"Dovish Signals     : {result['dovish_signals_found']}")
