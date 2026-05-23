class Planner:

    def __init__(self):
        self.tasks = [
            "summary",
            "contributions",
            "method",
            "dataset",
            "limitations",
            "future_work",
            "equations",
            "implementation",
            "research_gaps"
        ]

    def get_tasks(self):
        return self.tasks