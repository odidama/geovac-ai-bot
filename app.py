import streamlit as st
import pandas as pd
import json

from agent import query_agent, create_agent


def decode_response(content: str) -> dict:
    """
        This function converts the string response from the model to a dictionary object.
        Args: response (str): response from the model
        Returns: dict: dictionary with response data
    """
    return json.loads(content)


def write_response(response_dict: dict):
    """
        Write a response from an agent to a Streamlit app.
        Args: response_dict: The response from the agent.
        Returns: None.
    """

    # is the response is an answer? i.e. normal text response
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # is the resp a bar graph
    if "bar" in response_dict:
        bar_data = response_dict["bar"]
        df = pd.DataFrame(bar_data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # is the response a line chart
    if "line" in response_dict:
        line_data = response_dict["line"]
        df = pd.DataFrame(line_data)
        df.set_index("columns", inplace=True)

    # is the response a table
    if "table" in response_dict:
        table_data = response_dict["table"]
        df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
        st.table(df)


st.title("Odidama AI. Chat with your data")

st.write("Please upload a csv file to begin...")

data = st.file_uploader("Upload a csv...")

query = st.text_area("Ask questions of your data...")

if st.button("Submit Query", type="primary"):
    # create an agent from the CSV file
    agent = create_agent(data)

    # query agent
    response = query_agent(agent=agent, query=query)

    # decode response
    decoded_response = decode_response(response)

    write_response(decoded_response)
