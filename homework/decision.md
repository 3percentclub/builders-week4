# Decision: The Three C's

One or two sentences each. Replace the text after each arrow.

## Level 1 (required)

**Complexity:** Is chunking a job for rules (code) or for a model? Why?
→

**Context:** Your fix repeats the header in every chunk. What does that cost (tokens, index size), and why is it worth it?
→

**Criticism:** `test_2` only checks length. What could a chunker do that passes every test but still makes retrieval worse? (Hint: where does it cut?)
→

## Level 2 (optional challenge)

**Why ranks, not scores:** Vector search returns cosine similarity (0 to 1) and BM25 returns scores like 7.4. Why does RRF ignore those scores and use only the rank?
→

**k:** What happens to the results when k is 0 vs 60? Which one rewards being #1 in a single list more?
→

## Level 3 (optional bonus)

**The bug:** What did search A (raw history) return, and why?
→

**The fix:** What did your model rewrite the follow-up into? Did the clerk end up recommending the right tape?
→
