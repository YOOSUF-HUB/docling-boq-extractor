import re

from app.schemas.boq import (
    ExtractedBOQ,
    ExtractedBOQItem,
    ExtractedBOQSection,
)


class BOQNormalizer:

    SECTION_PATTERN = re.compile(
        r"^(\d+)\.\s*(.+)$"
    )

    ITEM_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)+)$"
    )

    def normalize(self, document) -> ExtractedBOQ:
        project_name = self._extract_project_name(document)

        sections = []

        for table in document.tables:
            sections.extend(
                self._extract_sections_from_table(
                    table,
                    document
                )
            )

        return ExtractedBOQ(
            project_name=project_name,
            title="BILL OF QUANTITIES",
            sections=sections,
        )

    def _extract_project_name(self, document):

        for text_item in document.texts:

            text = text_item.text.strip()

            match = re.search(
                r"PROJECT:\s*(.*?)\s+BILL\s+OF\s+QUANTITIES",
                text,
                re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        return None

    def _extract_sections_from_table(
        self,
        table,
        document
    ):
        dataframe = table.export_to_dataframe(
            doc=document
        )

        sections = []
        current_section = None

        for _, row in dataframe.iterrows():

            values = [
                str(value).strip()
                if value is not None
                else ""
                for value in row.tolist()
            ]

            if not any(values):
                continue

            first_value = values[0]

            # Section row
            if self._is_section(first_value):

                code, name = self._parse_section(
                    first_value
                )

                current_section = ExtractedBOQSection(
                    code=code,
                    name=name,
                )

                sections.append(current_section)

                continue

            # Item row
            if self._is_item(first_value):

                if current_section is None:
                    continue

                item = ExtractedBOQItem(
                    code=first_value,
                    description=values[1],
                    unit=values[2],
                    quantity=self._parse_quantity(
                        values[3]
                    ),
                )

                current_section.items.append(item)

        return sections

    def _is_section(self, value: str) -> bool:
        return bool(
            re.match(
                r"^\d+\.(?!\d)\s*.+$",
                value
            )
        )

    def _parse_section(self, value: str):

        match = re.match(
            r"^(\d+)\.(?!\d)\s*(.+)$",
            value
        )

        if not match:
            return None, value

        return (
            match.group(1),
            match.group(2).strip()
        )

    def _is_item(self, value: str) -> bool:
        return bool(
            self.ITEM_PATTERN.match(value)
        )

    def _parse_quantity(self, value: str):
        try:
            return float(
                value.replace(",", "")
            )
        except (ValueError, AttributeError):
            return None