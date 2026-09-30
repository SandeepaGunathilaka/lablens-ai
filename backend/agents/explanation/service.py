from .agent import ExplanationAgent
from .models import ExplanationRequest


class ExplanationService:

    def __init__(self):
        self.agent = ExplanationAgent()

    def generate_explanation(
        self,
        request: ExplanationRequest
    ):
        return self.agent.explain(request)