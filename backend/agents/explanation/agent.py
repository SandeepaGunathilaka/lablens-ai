import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .models import ExplanationRequest
from .prompts import SYSTEM_PROMPT


load_dotenv()


class ExplanationAgent:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(api_key=api_key)

    def explain(self, request: ExplanationRequest) -> str:

        findings = [
            finding.model_dump()
            for finding in request.findings
        ]

        user_prompt = f"""
Explain the following medical laboratory findings.

Findings:
{json.dumps(findings, indent=2)}

Patient context:
{request.patient_context or "No additional context provided."}

Provide an educational explanation for each finding.
Do not diagnose the patient.
Do not recommend treatment.
Do not invent missing information.
"""

        response = self.client.responses.create(
            model="gpt-5.6-luna",
            instructions=SYSTEM_PROMPT,
            input=user_prompt
        )

        return response.output_text