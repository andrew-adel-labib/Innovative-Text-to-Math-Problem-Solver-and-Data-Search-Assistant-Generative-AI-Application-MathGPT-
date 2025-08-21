import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(page_title="Text To MAth Problem Solver and Data Serach Assistant", page_icon="🧮")
st.title("Innovative Text To Math Problem Solver and Data Serach Assistant uing Google Gemma 2 (Maths GPT)")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please, Add your Groq API Key to continue!!")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find information on the mentioned topics."
)

math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only mathematical expressions need to be provided."
)

template = """
You are an agent tasked with solving users' mathematical questions. Logically work through the problem, provide a detailed explanation, and present the answer in a clear, point-wise format for the question below.
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a math chatbot that can answer all your math questions."}
    ]

for msg in  st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area(label="Enter Your Question Here", value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then, I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have in the end?")

if st.button("Find my Answer"):
    if question:
        with st.spinner("Generating Response ....."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(st.session_state.messages, callbacks=[st_callback_handler])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Response: ")
            st.success(response)

    else:
        st.warning("Please, Enter the question!!")