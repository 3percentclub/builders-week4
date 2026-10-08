# Instructor notes (Maria)

Retested on Oct 8, 2026, with **no API key** (local embeddings), and every code cell ran without errors. The data is a fixed file, so class-night numbers should match these closely.

## What the test run printed

| Experiment | Result on Oct 8 (no key) |
|---|---|
| 1. Chunk size | 512 → 38 chunks, 0 orphans. 128 → 106 chunks, **68 orphans**. Title + ending in the top 5: **512 = 5/5, 128 = 0/5**. The #1 chunk at 128 is Halloween's ending ("When Loomis looks down at the lawn, the body is gone"), with no title anywhere in it. |
| 2. Retrieval mode | `1313` (Oculus): **vector = not in top 10**, BM25 = #1, hybrid = #5. `Pleasence`: **vector = not in top 10**, BM25 = #1, hybrid = #3. |
| 3. Re-ranking | "A slasher where the final girl's friends die while she babysits": Halloween at **vector #8 → hybrid #1 → FlashRank #1**. With local embeddings, hybrid already fixes this one, so the story is "BM25 rescued it," and FlashRank keeps it at #1. |
| 4. Memory | Comparison now runs through the full hybrid + FlashRank pipeline. Raw history: top 3 = Blair Witch, Paranormal Activity, The Exorcist (**no Train to Busan**). Standalone question: **Train to Busan at #1**. The chat engine cell needs an LLM key; without one it skips. |

## Talking points the run gave us

- **Exp 1:** read the orphan chunk out loud and ask the room which movie it is. Horror fans will guess, but the model can't.
- **Exp 2:** popular names like "Mia Goth" or "Sosie Bacon" embed fine, and vector search finds them. What vector search misses are **meaningless tokens**: tape numbers and rare surnames.
- **Exp 3:** the re-ranker is the larger `ms-marco-MiniLM-L-12-v2`. The default tiny FlashRank model did NOT fix this query when I tested it, which is a good "model choice matters" aside.
- **Memory:** the chat engine is honest. It says Train to Busan isn't found footage, then recommends it anyway. That's grounding working as intended.

## Before class

1. Open the repo in Colab via File → Open notebook → GitHub and run all cells once, both without a key and with one (to see the chat engine).
2. At the lab slide, Ruth drops the repo link + "File → Save a copy in Drive" in Zoom chat and WhatsApp.
3. No key is required. Fellows who have any key (OpenRouter, Groq, Gemini, OpenAI...) paste it at the Block 1 prompt to unlock the last two cells. Everyone else presses Enter.

## Common room issues

- "LLM check failed": bad or empty-credit key. The lab continues without an LLM, so it's not a blocker.
- Colab secret not picked up: the secret's **Notebook access** toggle is off.
- `NameError`: someone skipped a cell. Tell them to use **Runtime → Run all**.
