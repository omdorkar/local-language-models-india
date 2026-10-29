# Language and Script Gaps in Local Language Models for Indian Users: summary for reviewers

**Om Dorkar**, independent research project (high-school student), October 2026. Full paper: [`paper.md`](paper.md).

## The question
If the same question is asked to an AI model in English, in an Indian language, or in an Indian language typed in English letters (for example "aap kaise hain"), does the model answer it equally well? And can simple software that runs on the same computer close any gap?

## Why it matters
Many people in India write to computers in their own language, often in English letters or mixed with English words. If AI tools answer them less accurately than English users, they get a worse service for reasons unrelated to what they asked. Small "local" models that run on a laptop or phone, with no internet and no cost per question, matter in India, but little is known about how well they handle Indian languages.

## What was done
* **Five open AI models** that fit on one laptop: Gemma 3 (Google), Qwen3 (Alibaba), Llama 3.1 (Meta), gpt-oss (OpenAI) and Sarvam-30B (Sarvam AI, an Indian company).
* **The same 300 reading-comprehension questions** from a public benchmark (Belebele), in English and **13 Indian languages**, and for Hindi, Bengali, Urdu and Nepali also in English letters. Two public knowledge-question sets (Global-MMLU) were used as a cross-check.
* **Two fixes** tested: converting English-letter text back to the native script (IndicXlit), and translating the question into English first (NLLB).
* **76,400 answers** in total, all produced **fully offline on one MacBook** (about 46 hours of computing). No cloud service, no paid service, no personal data and no human participants.
* **Five hypotheses (H1 to H5)** with named statistical tests (exact McNemar with Holm correction). All five results are reported, including the ones that did not come out as expected. Work that was not done and implementation details are listed in the paper (Section 3.8).

## Key findings
1. **Every Indian language was answered less accurately than English**, in all 65 model–language combinations, by 7.0 to 35.7 percentage points (average 18.8).
2. **English letters usually made it worse.** For Bengali, Urdu and Nepali, writing in English letters lowered accuracy further in 14 of 15 cases, by up to 31.3 points. Hindi was the exception: only one model was significantly worse on Hindi in English letters.
3. **Converting back to the native script often helped**: accuracy rose significantly in 10 of 20 cases, by up to 32.7 points.
4. **Translating to English first did not help as expected**: it helped significantly in only 3 of 65 cases and made results significantly worse in 9. The number of "tokens" (word pieces) a language needs did not predict how much accuracy it lost.

In exploratory analyses (not among the five hypotheses), the Indian-built model had a smaller gap between English and Indian languages than the others (16.0 vs 19.5 points on average), partly because its English score was also the lowest of the five.

## What was built and released
* Open-source code (MIT licence) with a single script, `reproduce.sh`, that repeats the whole study, and an analysis that rebuilds every table and figure from the saved answers.
* The question samples, the machine-made versions of the questions, and all 76,400 model answers, with a datasheet and an ethics statement.
* A local results dashboard (`dashboard/app.py`).

## Limitations
The results cover five compressed models on one multiple-choice reading task with 300 questions, so they may not hold for other models, open-ended conversation, or real users' writing. Each fix was tested with one specific tool, and the Indian English and Hinglish versions were machine-made approximations, not real people's writing.

## Where to look
* Paper: [`paper/paper.md`](paper.md)
* All results: [`reports/results/summary.md`](../reports/results/summary.md) · Figures: [`reports/figures/`](../reports/figures/)
