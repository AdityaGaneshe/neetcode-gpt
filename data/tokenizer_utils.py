from typing import List, Dict

class Solution:
    def _tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        max_len = max((len(k) for k in vocab), default=1)

        while i < n:
            matched = None
            # Try the longest possible substring first
            for end in range(min(n, i + max_len), i, -1):
                candidate = text[i:end]
                if candidate in vocab:
                    matched = candidate
                    break
            if matched is None:
                # No match: consume a single character
                matched = text[i]
            tokens.append(matched)
            i += len(matched)

        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        return [self._tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        return len(self._tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        words = text.split(" ")
        words = [w for w in words if w]
        if not words:
            return 0.0
        return round(self.count_tokens(text, vocab) / len(words), 4)