from analysis.summary_agent import generate_summary
from analysis.contributions_agent import generate_contributions
from analysis.method_agent import generate_method_explanation
from analysis.dataset_agent import generate_dataset_explanation
from analysis.limitations_agent import generate_limitations
from analysis.future_work_agent import generate_future_work
from analysis.equation_agent import generate_equation_explanation
from analysis.implementation_agent import generate_implementation_ideas
from analysis.research_gap_agent import generate_research_gaps


def execute_tasks(retriever, planner, memory):

    for task in planner.get_tasks():

        if task == "summary":
            memory.store(task, generate_summary(retriever))

        elif task == "contributions":
            memory.store(task, generate_contributions(retriever))

        elif task == "method":
            memory.store(task, generate_method_explanation(retriever))

        elif task == "dataset":
            memory.store(task, generate_dataset_explanation(retriever))

        elif task == "limitations":
            memory.store(task, generate_limitations(retriever))

        elif task == "future_work":
            memory.store(task, generate_future_work(retriever))

        elif task == "equations":
            memory.store(task, generate_equation_explanation(retriever))

        elif task == "implementation":
            memory.store(task, generate_implementation_ideas(retriever))

        elif task == "research_gaps":
            memory.store(task, generate_research_gaps(retriever))

    return memory.get_all()