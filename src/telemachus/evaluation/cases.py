from dataclasses import dataclass


@dataclass
class EvaluationCase:
    query: str
    relevant: set[str]
    task_category: str | None


EVAL_CASES: list[EvaluationCase] = [
    EvaluationCase(
        query="dataset for training robots to perform physical manipulation tasks",
        relevant={
            "genrobot2025/10Kh-RealOmin-OpenData",
            "XDOF/ABC-130k",
            "InternRobotics/InternData-A1",
            "cadene/droid",
        },
        task_category="robotics",
    ),
    EvaluationCase(
        query="English dataset for training a text classification model",
        relevant={
            "nyu-mll/glue",
            "aps/super_glue",
            "stanfordnlp/imdb",
        },
        task_category="text-classification",
    ),
    EvaluationCase(
        query="English dataset for training a question-answering model",
        relevant={
            "allenai/ai2_arc",
            "rajpurkar/squad",
        },
        task_category="question-answering",
    ),
    EvaluationCase(
        query="English legal text corpus suitable for training or fine-tuning a language model",
        relevant={
            "pile-of-law/pile-of-law",
            "mratanusarkar/Indian-Laws",
            "a2aj/canadian-case-law",
            "HFforLegal/case-law",
        },
        task_category=None,
    ),
    EvaluationCase(
        query="English medical text dataset suitable for training or fine-tuning a language model",
        relevant={
            "lavita/medical-qa-datasets",
            "medalpaca/medical_meadow_medqa",
            "medalpaca/medical_meadow_medical_flashcards",
            "medalpaca/medical_meadow_wikidoc",
            "FreedomIntelligence/medical-o1-reasoning-SFT",
        },
        task_category=None,
    ),
    EvaluationCase(
        query="English financial text corpus suitable for training or fine-tuning a language model",
        relevant={
            "artefactory/Argimi-Ardian-Finance-10k-text",
            "vidore/vidore_v3_finance_en",
            "gbharti/finance-alpaca",
        },
        task_category=None,
    ),
]
