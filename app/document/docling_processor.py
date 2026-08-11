from pathlib import Path

from docling.document_converter import DocumentConverter


class DoclingProcessor:

    def __init__(self):
        self.converter = DocumentConverter()

    def process(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        result = self.converter.convert(str(path))

        return result.document