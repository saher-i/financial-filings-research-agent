# Evaluation pipeline

This package contains the development answer runner and retrieval and generation metrics. It is separate from the answering agent.

## Generation

In developer mode, open `/generation-evaluation` after generating answers on `/benchmark-evaluation`. **Test 2 questions** scores q_001 (SQL) and q_012 (SQL and PDF) first. The ten-question button unlocks only after both official metrics return scores for both pilot questions. Neither button regenerates answers. The pipeline uses RAGAS `Faithfulness` and `AnswerRelevancy` from `ragas.metrics.collections`, with Fireworks GLM-5.3 as the judge for claim checks and question generation. Faithfulness receives gathered SQL/PDF contexts, and Response Relevancy uses the original question, answer, and Fireworks Nomic embeddings. SQL contexts retain the original rows and add exact billion-dollar renderings so the judge can compare stated units. The page shows the official Faithfulness claim verdicts and reasons. Saved answer citations and each claim's evidence control open the original SQL result or PDF page; RAGAS does not assign exact source IDs to individual claims, so claim controls list the answer's cited sources. New runs also record the generated Response Relevancy questions and their embedding similarities for inspection; these traces do not alter RAGAS's score. The golden answer is not sent to either metric. Per-question scores and errors are saved under `evaluation/generation_runs/` in separate pilot and full files; older custom-score files are never shown as RAGAS results.

## Retrieval

Open `/retrieval-evaluation` after generating benchmark answers. The two-question pilot or ten-question full run loads each saved answer and scores only its cited SQL results and PDF chunks, in first-citation order. `retrieval.py` calls the official RAGAS `ContextPrecision` and `ContextRecall` scorers with those contexts and the supplied reference answer. A Fireworks-hosted LLM judges source usefulness and reference-claim attribution; RAGAS calculates the two scores. No human passage labels or old target phrases are used. The standard RAGAS prompt retains each verdict's explanation.

These are cited-evidence scores, not search-quality scores: uncited retrieved passages are excluded, and citation order replaces retrieval rank. Both official RAGAS scores remain LLM estimates. A larger human-labeled set is needed for independent judge calibration. Unavailable judgments have null scores and are excluded from averages.
