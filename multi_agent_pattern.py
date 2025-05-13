import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


load_dotenv()

model = "gpt-4o-mini"

api_key = os.environ["OPENAI_API_KEY"]


config_list = [
    {
        "model": "llama3.2:3b",  # add your own model here
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
]

# == Uncomment the following line to use OpenAI API ==
llm_config = {"model": model, "temperature": 0.0, "api_key": api_key}

# == Using the local Ollama model ==
# llm_config = {"config_list": config_list, "temperature": 0.0}


class ResearchAnalysisSystem:
    def __init__(self):
        self.llm_config = llm_config

        self.code_execution_config = {
            "work_dir": "research_output",
            "use_docker": False,
        }

        self.setup_agents()
        self.setup_group_chat()

    def setup_agents(self):
        # User Proxy for coordination
        self.user_proxy = UserProxyAgent(
            name="admin",
            human_input_mode="NEVER",
            code_execution_config=self.code_execution_config,
            system_message="Admin coordinating research analysis.",
        )

        # Specialized Agents
        self.research_agent = AssistantAgent(
            name="researcher",
            llm_config=self.llm_config,
            system_message="""Find and analyze research papers. Focus on:
            1. Identifying relevant papers
            2. Extracting key findings
            3. Categorizing applications""",
        )

        self.data_analyst = AssistantAgent(
            name="analyst",
            llm_config=self.llm_config,
            system_message="""Analyze research data to:
            1. Extract metrics
            2. Generate visualizations
            3. Identify trends""",
        )

        self.reviewer = AssistantAgent(
            name="reviewer",
            llm_config=self.llm_config,
            system_message="""Review findings for:
            1. Accuracy
            2. Completeness
            3. Relevance""",
        )

        self.content_writer = AssistantAgent(
            name="writer",
            llm_config=self.llm_config,
            system_message="""Synthesize findings into clear reports with:
            1. Executive summaries
            2. Detailed analysis
            3. Recommendations""",
        )

    def setup_group_chat(self):
        # Create agent group
        self.agents = [
            self.user_proxy,
            self.research_agent,
            self.data_analyst,
            self.reviewer,
            self.content_writer,
        ]

        # Initialize group chat
        self.group_chat = GroupChat(agents=self.agents, messages=[], max_round=50)

        # Set up manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat, llm_config=self.llm_config
        )

    def research_pipeline(self, topic):
        # Initialize research tasks
        tasks = [
            f"Research papers on {topic}",
            "Extract and categorize applications",
            "Generate visualization of findings",
            "Prepare comprehensive report",
        ]

        results = []
        for task in tasks:
            # Each task is processed by the group
            result = self.user_proxy.initiate_chat(
                self.manager, message=task, summary_method="last_msg"
            )
            results.append(result)

        return results

    def analyze_article(self, article_content):
        # Article analysis pipeline
        analysis_tasks = [
            f"Analyze structure and coherence: {article_content}",
            "Review style and tone",
            "Verify factual accuracy",
            "Provide improvement suggestions",
            "Generate final publication readiness report",
        ]

        analysis_results = []
        for task in analysis_tasks:
            result = self.user_proxy.initiate_chat(
                self.manager, message=task, summary_method="last_msg"
            )
            analysis_results.append(result)

        return analysis_results


def main():
    # Initialize system
    system = ResearchAnalysisSystem()

    # Example usage
    topic = "machine learning in healthcare"
    research_results = system.research_pipeline(topic)

    # Article analysis
    with open("article.txt", "r") as file:
        article_content = file.read()

    analysis_results = system.analyze_article(article_content)

    # Print final results
    print("\nResearch Analysis Results:")
    for i, result in enumerate(research_results):
        print(f"\nTask {i+1} Results:")
        print(result)

    print("\nArticle Analysis Results:")
    for i, result in enumerate(analysis_results):
        print(f"\nAnalysis Task {i+1} Results:")
        print(result)


if __name__ == "__main__":
    main()
