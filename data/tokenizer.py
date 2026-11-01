from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            # Count adjacent pairs
            counts = Counter()
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += 1

            if not counts:
                break

            # Highest frequency; ties broken by lexicographically smallest pair
            best = min(counts, key=lambda p: (-counts[p], p))
            a, b = best
            merges.append([a, b])

            # Merge non-overlapping occurrences, left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges