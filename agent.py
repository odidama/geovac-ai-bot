from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

import environ

env = environ.Env()
environ.Env.read_env()
api_key = env("GOOGLE_API_KEY")


# print(api_key)

def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).
    Args: filename: The path to the CSV file that contains the data.
    Returns: An agent that can access and use the LLM along with the supplied file
    """

    llm = GoogleGenerativeAI(model="gemini-1.5-flash", max_retries=2,
                             api_key=api_key, temperature=0)
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)


def query_agent(agent, query):
    """
    Query an agent and return the response as a string.
    Args:
        agent: The agent to be queried.
        query: interaction with the LLM.
    Returns:
        The response from the LLM through the agent as str.
    """

    prompt = (
            """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query:
        """
            + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)
    return response.__str__()
