from pydantic import BaseModel, Field
from typing import List, Optional

class MCQAnswer(BaseModel):
    """Model for MCQ answers"""
    selected_option: str = Field(description="The selected option letter (A, B, C, D)")
    explanation: str = Field(description="Detailed explanation for the selected answer")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class TrueFalseAnswer(BaseModel):
    """Model for True/False answers"""
    answer: bool = Field(description="The answer (True or False)")
    explanation: str = Field(description="Detailed explanation for the answer")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class ShortAnswer(BaseModel):
    """Model for short answers"""
    answer: str = Field(description="The answer text")
    explanation: str = Field(description="Detailed explanation for the answer")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class LLMResponse(BaseModel):
    """Model for LLM responses"""
    question_id: str = Field(description="The ID of the question")
    question_text: str = Field(description="The text of the question")
    question_type: str = Field(description="The type of question (MCQ, True/False, Short Answer)")
    mcq_answer: Optional[MCQAnswer] = Field(None, description="MCQ answer if applicable")
    true_false_answer: Optional[TrueFalseAnswer] = Field(None, description="True/False answer if applicable")
    short_answer: Optional[ShortAnswer] = Field(None, description="Short answer if applicable")
    model_name: str = Field(description="The name of the model that generated the response")
    timestamp: str = Field(description="Timestamp of the response")
    version: Optional[str] = Field(None, description="Version of the question if applicable") 