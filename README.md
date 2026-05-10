# CoMAT & Shapley Value Analysis — NLP Coursework

**University of Edinburgh** · Natural Language Understanding, Generation and Machine Translation (NLU+) · Spring 2025 (exchange)

Evaluation of GPT-4o-mini on the MMLU-Redux College Mathematics benchmark using the CoMAT (Chain of Mathematical Thought) prompting strategy, followed by Shapley value analysis to quantify the contribution of each reasoning step to model accuracy.

---

## Assignment Overview

The coursework had four parts:

1. **Q1 — Formalising Questions**: Manually apply the CoMAT 7-step symbolic reasoning framework to two math problems, then critically evaluate its utility and propose extensions.
2. **Q2 — Shapley Value Analysis**: Implement the Shapley value pipeline in code, run the CoMAT evaluation against the MMLU-Redux College Mathematics dataset, and interpret the results.
3. **Q3 — Failure Analysis**: Find a question where CoMAT fails, diagnose the failure step, and propose and test a fix.
4. **Q4 — LLM-Assisted Proof**: Prove the equivalence of two Shapley value formulations, using an LLM as a collaborator, and critically reflect on the experience.

## Implementation

The skeleton code provided the dataset processing loop (`mmlu_redux.py`) and the Shapley computation scaffold. Core implementations:

- **`utils.py`** (`predict_gpt`): OpenAI API call using GPT-4o-mini, temperature 0.0, max_tokens 2000 — deterministic inference for reproducibility
- **`CoMAT_Instruction.py`** (`INSTRUCTION`): 7-step CoMAT system prompt — predicate/function definition → logical rule parsing → fact extraction → symbolic question parse → step-by-step solve → answer derivation → option matching
- **`shapley_value_evaluation.py`**:
  - `get_missing_steps`: maps per-row `step{n}_missing` flags to a sorted tuple of absent steps
  - `compute_v_S`: computes v(S) = mean accuracy over all questions sharing the same missing-step subset S
  - `compute_marginal_contributions`: iterates all permutations π of steps, accumulates Δᵢ(π) = v(S⁻ᵢ ∪ {i}) − v(S⁻ᵢ) per step
  - `compute_shapley_values`: divides accumulated Δ sums by the count of valid permutations

---

## Written Report

**Q2 — Shapley Value Analysis**: Computed Shapley values for each of the 4 CoMAT reasoning steps (Step 2 "Structural Logic Translation" highest at 0.0407; Step 4 lowest at 0.0192). Identified a non-obvious step interdependency: removing Step 1 alone hurt accuracy more than removing both Steps 1 and 2, since Step 2 without Step 1's variable definitions constructs semantically ungrounded equations that actively mislead the model.

**Q3 — Failure Analysis & Fix** (full marks): Diagnosed a modular arithmetic failure where CoMAT identified 3x + 7y ≡ 0 mod 11 but failed to derive the key equivalence x ≡ 5y mod 11 via modular inversion. Proposed **Step 2.5 "Symbolic Equivalence Derivation"** — a new mandatory step inserted between Steps 2 and 3 that derives alternative symbolic forms before solving. Validated empirically: accuracy improved from **75.76% → 76.77%** on the full dataset.

**Q4 — LLM-Assisted Proof** (full marks): Proved the equivalence of the permutation-based and subset-based Shapley value formulations. Key insight: each subset S appears z = |S|! · (n−1−|S|)! times across all permutations. Developed the proof independently, then used GPT-4o to explore alternatives — the LLM's combinatorial derivation of z was cleaner and was adopted. Also compared DeepSeek, Gemini, and GPT-4o-mini with CoMAT on the same proof task.

---

## Usage

```bash
pip install -r requirements.txt
# add OPENAI_API_KEY=<your_key> to code_question/.env

cd code_question

# Run CoMAT evaluation (GPT-4o-mini on College Mathematics)
python main.py --dataset mmlu-redux-college_mathematics --method comat --model gpt

# Run Shapley value analysis on pre-computed step data
python shapley_value_evaluation.py
```

---

## Stack

Python 3 · OpenAI API (GPT-4o-mini) · pandas · NumPy · MMLU-Redux College Mathematics (148 questions)
