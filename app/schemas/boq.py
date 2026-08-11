from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# NORMALIZED BOQ
# Used by the deterministic Docling normalization pipeline.
# ============================================================

class ExtractedBOQItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str | None = None
    description: str | None = None
    unit: str | None = None
    quantity: float | None = None


class ExtractedBOQSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str | None = None
    name: str
    items: list[ExtractedBOQItem] = Field(
        default_factory=list
    )


class ExtractedBOQ(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str | None = None
    title: str | None = None
    sections: list[ExtractedBOQSection] = Field(
        default_factory=list
    )


# ============================================================
# FINAL BOQ
# Used by Groq Structured Outputs.
# ============================================================

class FinalBOQItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str | None
    description: str | None
    unit: str | None
    quantity: float | None


class FinalBOQSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str | None
    name: str
    items: list[FinalBOQItem]


class FinalBOQ(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str | None
    title: str | None
    sections: list[FinalBOQSection]