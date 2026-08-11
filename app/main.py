import json
from pathlib import Path

from app.document.docling_processor import DoclingProcessor
from app.normalization.boq_normalizer import BOQNormalizer


def main():

    processor = DoclingProcessor()

    document = processor.process(
        "data/input/sample_boq_table_valuesmissing.pdf"
    )

    normalizer = BOQNormalizer()

    boq = normalizer.normalize(document)

    output_path = Path(
        "data/output/normalized_boq.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            boq.model_dump(),
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Normalized BOQ saved to: {output_path}"
    )

    print("\n========== NORMALIZED BOQ ==========\n")

    print(
        json.dumps(
            boq.model_dump(),
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()