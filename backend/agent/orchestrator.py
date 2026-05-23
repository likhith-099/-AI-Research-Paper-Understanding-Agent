from analysis.summary_agent import generate_summary
from analysis.contributions_agent import generate_contributions
from analysis.method_agent import generate_method_explanation
from analysis.dataset_agent import generate_dataset_explanation
from analysis.limitations_agent import generate_limitations
from analysis.future_work_agent import generate_future_work
from analysis.equation_agent import generate_equation_explanation
from analysis.implementation_agent import generate_implementation_ideas
from analysis.research_gap_agent import generate_research_gaps
from analysis.llm_client import API_KEY_ERROR_MESSAGE


def _safe_analysis(func, retriever, default_message=None):
    """Safely call analysis functions and return user-friendly error messages."""
    try:
        return func(retriever)
    except ValueError as e:
        if "API key" in str(e) or "GROQ_API_KEY" in str(e) or "GROQ_API_KEY" in str(e):
            return default_message or API_KEY_ERROR_MESSAGE
        raise
    except ModuleNotFoundError as e:
        return str(e)
    except RuntimeError as e:
        return str(e)


def analyze_paper(retriever):
    report = {}

    report["summary"] = _safe_analysis(generate_summary, retriever)
    report["contributions"] = _safe_analysis(generate_contributions, retriever)
    report["method"] = _safe_analysis(generate_method_explanation, retriever)
    report["dataset"] = _safe_analysis(generate_dataset_explanation, retriever)
    report["limitations"] = _safe_analysis(generate_limitations, retriever)
    report["future_work"] = _safe_analysis(generate_future_work, retriever)
    report["equations"] = _safe_analysis(generate_equation_explanation, retriever)
    report["implementation"] = _safe_analysis(generate_implementation_ideas, retriever)
    report["research_gaps"] = _safe_analysis(generate_research_gaps, retriever)

    return report
