# Instructor notes (Maria)

I tested this end to end on Oct 7, 2026, and every code cell ran without errors. The data is a fixed file, so class-night numbers should match these closely.

## What the test run printed

| Experiment | Result on Oct 7 |
|---|---|
| 1. Chunk size | 512 → 38 chunks, 0 orphans. 128 → 106 chunks, **68 orphans**. Title + ending in the top 5: **512 = 5/5, 128 = 0/5**. The #1 chunk at 128 is Halloween's ending ("When Loomis looks down at the lawn, the body is gone"), with no title anywhere in it. |
| 2. Retrieval mode | `1313` (Oculus): **vector = not in top 10**, BM25 = #1, hybrid = #4. `Pleasence`: **vector = not in top 10**, BM25 = #1, hybrid = #3. |
| 3. Re-ranking | "A slasher where the final girl's friends die while she babysits": Halloween at **vector #7 → hybrid #2 → FlashRank #1**. |
| 4. Memory | Raw history as the query: top 3 = Blair Witch, Paranormal Activity, Cabin in the Woods (**no Train to Busan**). Standalone question: **Train to Busan at #1**. The chat engine rewrote the follow-up as "Can you recommend a found-footage movie set on a train?" |

## Talking points the run gave us

- **Exp 1:** read the orphan chunk out loud and ask the room which movie it is. Horror fans will guess, but the model can't.
- **Exp 2:** popular names like "Mia Goth" or "Sosie Bacon" embed fine, and vector search finds them. What vector search misses are **meaningless tokens**: tape numbers and rare surnames.
- **Exp 3:** the re-ranker is the larger `ms-marco-MiniLM-L-12-v2`. The default tiny FlashRank model did NOT fix this query when I tested it, which is a good "model choice matters" aside.
- **Memory:** the chat engine is honest. It says Train to Busan isn't found footage, then recommends it anyway. That's grounding working as intended.

## Before class

1. Open the repo in Colab via File → Open notebook → GitHub and run all cells once with your key.
2. At the lab slide, Ruth drops the repo link + "File → Save a copy in Drive" in Zoom chat and WhatsApp.
3. Fellows without an OpenAI key or credit pair with a neighbor. Each run costs under $0.05.

## Common room issues

- 401 at the key check: in Colab, the secret's **Notebook access** toggle is off.
- `NameError`: someone skipped a cell. Tell them to use **Runtime → Run all**.
