"""The one prompt template used for every condition. Instructions stay in English so only the
passage, question and options change between languages and scripts. Its hash is stored with every answer."""

import hashlib

LETTERS = "ABCD"

BELEBELE_TEMPLATE = """Read the passage and answer the multiple-choice question. Reply with JSON only, in the form {{"answer": "<letter>"}}, where <letter> is A, B, C or D.

Passage:
{passage}

Question:
{question}

A. {a}
B. {b}
C. {c}
D. {d}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(LETTERS)}},
    "required": ["answer"],
}

TEMPLATE_SHA = hashlib.sha256(BELEBELE_TEMPLATE.encode()).hexdigest()[:12]


def belebele_prompt(row) -> str:
    return BELEBELE_TEMPLATE.format(
        passage=row["flores_passage"], question=row["question"],
        a=row["mc_answer1"], b=row["mc_answer2"], c=row["mc_answer3"], d=row["mc_answer4"],
    )


def belebele_gold(row) -> str:
    return LETTERS[int(row["correct_answer_num"]) - 1]


# Global-MMLU (no passage). Same English instruction style and JSON answer format as the Belebele template.
GMMLU_TEMPLATE = """Answer the multiple-choice question. Reply with JSON only, in the form {{"answer": "<letter>"}}, where <letter> is A, B, C or D.

Question:
{question}

A. {a}
B. {b}
C. {c}
D. {d}"""

GMMLU_TEMPLATE_SHA = hashlib.sha256(GMMLU_TEMPLATE.encode()).hexdigest()[:12]


def gmmlu_prompt(row) -> str:
    return GMMLU_TEMPLATE.format(
        question=row["question"], a=row["option_a"], b=row["option_b"], c=row["option_c"], d=row["option_d"],
    )


def gmmlu_gold(row) -> str:
    return str(row["answer"]).strip().upper()
