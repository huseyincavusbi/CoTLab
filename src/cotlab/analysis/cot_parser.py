"""CoT Parser for extracting and analyzing reasoning steps."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """A single step in chain of thought reasoning."""
    index: int
    text: str
    is_claim: bool = False
    is_conclusion: bool = False


class CoTParser:
    """
    Extract structure from Chain of Thought outputs.
    
    Parses model outputs to identify:
    - Numbered reasoning steps
    - Factual claims
    - Hedging/uncertainty language
    - Final conclusions
    """
    
    # Patterns for step extraction
    STEP_PATTERNS = [
        r"(?:^|\n)\s*(\d+)[.):]\s*(.+?)(?=\n\s*\d+[.):)]|\n\n|$)",  # 1. Step
        r"(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\n|$)",  # Bullet points
        r"(?:^|\n)\s*(First|Second|Third|Then|Next|Finally)[,:]?\s*(.+?)(?=\n|$)",  # Word numbered
    ]
    
    # Hedging indicators
    HEDGING_WORDS = [
        "might", "could", "possibly", "perhaps", "maybe", 
        "uncertain", "unsure", "likely", "probably", "appears",
        "seems", "suggests", "may", "I think", "I believe",
        "not sure", "unclear", "would guess"
    ]
    
    # Confidence indicators
    CONFIDENCE_WORDS = [
        "definitely", "certainly", "clearly", "obviously", 
        "must be", "undoubtedly", "without doubt", "absolutely",
        "100%", "confident", "sure"
    ]
    
    # Conclusion markers
    CONCLUSION_MARKERS = [
        "therefore", "thus", "so", "hence", "consequently",
        "in conclusion", "final answer", "the answer is",
        "this means", "we can conclude"
    ]
    
    def extract_steps(self, cot_text: str) -> List[ReasoningStep]:
        """
        Parse numbered/bulleted reasoning steps from CoT.
        
        Args:
            cot_text: Raw CoT output
            
        Returns:
            List of ReasoningStep objects
        """
        steps = []
        
        # Try numbered pattern first
        numbered = re.findall(
            r"(?:^|\n)\s*(\d+)[.):]\s*(.+?)(?=\n\s*\d+[.):)]|\n\n|$)",
            cot_text,
            re.DOTALL
        )
        
        if numbered:
            for idx, (num, text) in enumerate(numbered):
                step = ReasoningStep(
                    index=idx,
                    text=text.strip(),
                    is_conclusion=self._is_conclusion(text)
                )
                steps.append(step)
        else:
            # Fall back to sentence-based splitting
            sentences = re.split(r'(?<=[.!?])\s+', cot_text)
            for idx, sent in enumerate(sentences):
                if sent.strip():
                    steps.append(ReasoningStep(
                        index=idx,
                        text=sent.strip(),
                        is_conclusion=self._is_conclusion(sent)
                    ))
        
        return steps
    
    def identify_claims(self, cot_text: str) -> List[Dict[str, Any]]:
        """
        Extract factual claims from reasoning.
        
        A claim is a statement that asserts something as true.
        
        Returns:
            List of dicts with 'text' and 'confidence' keys
        """
        claims = []
        sentences = re.split(r'(?<=[.!?])\s+', cot_text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Skip questions
            if sent.endswith('?'):
                continue
            
            # Check if it's a claim vs procedural text
            is_claim = any([
                " is " in sent.lower(),
                " are " in sent.lower(),
                " has " in sent.lower(),
                " have " in sent.lower(),
                " indicates " in sent.lower(),
                " suggests " in sent.lower(),
                " shows " in sent.lower(),
            ])
            
            if is_claim:
                confidence = self._estimate_confidence(sent)
                claims.append({
                    "text": sent,
                    "confidence": confidence,
                    "has_hedging": confidence < 0.5
                })
        
        return claims
    
    def detect_hedging(self, cot_text: str) -> float:
        """
        Measure uncertainty expressions in CoT.
        
        Returns:
            Score from 0 (no hedging) to 1 (heavy hedging)
        """
        text_lower = cot_text.lower()
        
        hedging_count = sum(
            1 for word in self.HEDGING_WORDS 
            if word.lower() in text_lower
        )
        confidence_count = sum(
            1 for word in self.CONFIDENCE_WORDS 
            if word.lower() in text_lower
        )
        
        total = hedging_count + confidence_count
        if total == 0:
            return 0.3  # Neutral default
        
        return hedging_count / total
    
    def extract_conclusion(self, cot_text: str) -> Optional[str]:
        """
        Extract the final conclusion/answer from CoT.
        
        Returns:
            The conclusion text, or None if not found
        """
        text_lower = cot_text.lower()
        
        for marker in self.CONCLUSION_MARKERS:
            pattern = rf"{marker}\s*[,:]?\s*(.+?)(?:\.|$)"
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                # Find the actual text in original case
                start = match.start(1)
                end = match.end(1)
                return cot_text[start:end].strip()
        
        # Fall back to last sentence
        sentences = re.split(r'(?<=[.!?])\s+', cot_text)
        if sentences:
            return sentences[-1].strip()
        
        return None
    
    def analyze(self, cot_text: str) -> Dict[str, Any]:
        """
        Full analysis of a CoT output.
        
        Returns:
            Dict with steps, claims, hedging score, and conclusion
        """
        return {
            "steps": self.extract_steps(cot_text),
            "claims": self.identify_claims(cot_text),
            "hedging_score": self.detect_hedging(cot_text),
            "conclusion": self.extract_conclusion(cot_text),
            "num_steps": len(self.extract_steps(cot_text)),
            "word_count": len(cot_text.split()),
        }
    
    def _is_conclusion(self, text: str) -> bool:
        """Check if text is a conclusion."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.CONCLUSION_MARKERS)
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence level of a claim."""
        text_lower = text.lower()
        
        has_hedging = any(w in text_lower for w in self.HEDGING_WORDS)
        has_confidence = any(w in text_lower for w in self.CONFIDENCE_WORDS)
        
        if has_hedging and not has_confidence:
            return 0.3
        elif has_confidence and not has_hedging:
            return 0.9
        elif has_hedging and has_confidence:
            return 0.5
        else:
            return 0.6  # Neutral
