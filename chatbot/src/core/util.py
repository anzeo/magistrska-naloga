from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, verbose=True)


def get_title_from_query(query):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Na podlagi uporabnikovega poziva, predlagaj naslov pogovora. Naslov naj bo kratek in jedrnat, ter brez robnih narekovajev."),
        ("human", "{query}")
    ])

    chain = prompt | chat_model | StrOutputParser()

    response = chain.invoke({"query": query})

    return response
