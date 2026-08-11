import json
import os

from dotenv import load_dotenv
from groq import Groq

from app.schemas.boq import FinalBOQ


load_dotenv()


class GroqBOQExtractor:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set"
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

    def extract(
        self,
        document_text: str,
        normalized_boq: dict,
    ):

        system_prompt = """
    You are a BOQ document reconstruction engine.

    Your task is to reconstruct a structured Bill of Quantities
    from the SOURCE DOCUMENT.

    SOURCE PRIORITY:

    1. SOURCE DOCUMENT is the authoritative source.
    2. DETERMINISTIC NORMALIZED BOQ is only an intermediate
    extraction attempt.
    3. The normalized BOQ may be incomplete or incorrect.
    4. Never assume that information missing from the normalized
    BOQ is missing from the source document.

    EXTRACTION RULES:

    1. Extract every section present in the source.
    2. Extract every BOQ item present in the source.
    3. Associate each item with its correct section.
    4. Preserve item codes exactly.
    5. Preserve descriptions from the source.
    6. Preserve quantities exactly.
    7. Extract units only when explicitly present.
    8. If a unit is not explicitly present, return null.
    9. If a quantity is not explicitly present, return null.
    10. Never infer a unit from another item.
    11. Never infer a quantity from another item.
    12. Do not invent sections or items.
    13. Do not add words to descriptions.
    14. Formatting inconsistencies in section codes should be
        normalized.

    For each item, inspect the SOURCE DOCUMENT itself.

    For example:

    "2.1 Blinding concrete 15 M3"

    means:

    code = "2.1"
    description = "Blinding concrete"
    quantity = 15
    unit = "M3"

    But:

    "2.2 Reinforced concrete foundations 50"

    means:

    code = "2.2"
    description = "Reinforced concrete foundations"
    quantity = 50
    unit = null

    Do not infer M3 for 2.2 just because other concrete
    items use M3.

    Return only the structured BOQ.
    """

        user_prompt = f"""
    SOURCE DOCUMENT
    ===============

    {document_text}


    DETERMINISTIC NORMALIZED BOQ
    =============================

    {json.dumps(normalized_boq, indent=2)}


    TASK
    ====

    Reconstruct the complete BOQ.

    Use the SOURCE DOCUMENT as the authoritative evidence.

    The deterministic normalized BOQ is only a preliminary
    attempt and may have missing sections, missing items,
    incorrect quantities, or incorrectly assigned units.

    Recover the complete structure from the source.
    """

        response = self.client.chat.completions.create(
            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            reasoning_effort="medium",

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "boq_structure",
                    "strict": True,
                    "schema": FinalBOQ.model_json_schema(),
                },
            },
        )

        content = response.choices[0].message.content

        return FinalBOQ.model_validate_json(
            content
        )