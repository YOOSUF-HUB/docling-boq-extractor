import json

from app.document.docling_processor import DoclingProcessor
from app.normalization.boq_normalizer import BOQNormalizer
from app.llm.groq_client import GroqBOQExtractor


PDF_PATH = "data/input/sample_boq_table_valuesmissing.pdf"

    
def main():

    # ==================================================
    # 1. DOCLING
    # ==================================================

    print("\n========== 1. DOCLING ==========\n")

    processor = DoclingProcessor()

    document = processor.process(PDF_PATH)

    document_text = document.export_to_markdown()

    print(document_text)

    # ==================================================
    # 2. DETERMINISTIC NORMALIZATION
    # ==================================================

    print(
        "\n========== 2. NORMALIZED BOQ ==========\n"
    )

    normalizer = BOQNormalizer()

    normalized_boq = normalizer.normalize(
        document
    )

    normalized_data = normalized_boq.model_dump()

    print(
        json.dumps(
            normalized_data,
            indent=2,
            ensure_ascii=False
        )
    )

    # ==================================================
    # 3. GROQ / GPT-OSS 120B
    # ==================================================

    print(
        "\n========== 3. GROQ / GPT-OSS 120B ==========\n"
    )

    extractor = GroqBOQExtractor()

    result = extractor.extract(
        document_text=document_text,
        normalized_boq=normalized_data,
    )

    # ==================================================
    # 4. FINAL RESULT
    # ==================================================

    print(
        "\n========== FINAL BOQ ==========\n"
    )

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()