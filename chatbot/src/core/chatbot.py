from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel, Field

from src.core.ai_act_summary import AI_ACT_SUMMARY
from src import sqlite_conn
from src.retriever.TFIDFRetriever import TFIDFRetriever

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, verbose=True)


class Top3Response(BaseModel):
    DocumentIDs: list[str] = Field(description="Seznam ID-jev treh najbolj relevantnih dokumentov")


class QueryHistoryRelationParser(BaseModel):
    Related: str = Field(description="Decision, whether the query is related to chat history")
    Reasoning: str = Field(description="Reasoning behind your decision")


class QueryClassificationParser(BaseModel):
    Relevance: str = Field(description="Izbrana kategorija")
    Reasoning: str = Field(description="Utemeljitev za izbiro kategorije")


class RelevantPartsParser(BaseModel):
    AnswerIncluded: str = Field(description='Odgovori izključno z "Da" ali "Ne"')
    RelevantParts: list[str] = Field(
        description='Seznam odlomkov iz dokumenta, ki podpirajo odgovor (ali prazen seznam [])')


class RelevantPassage(BaseModel):
    id: str
    text: list[str]


class RAGAnswerParser(BaseModel):
    Answer: str = Field(description='Odgovor pridobljen iz konteksta')
    RelevantParts: list[RelevantPassage] = Field(
        description='Seznam odlomkov iz dokumenta, ki podpirajo odgovor (ali prazen seznam [])')


class AnswerValidationParser(BaseModel):
    AnswerValid: str = Field(description="Odločitev ali je odgovor zadosten glede na zastavljeno vprašanje")
    Reasoning: str = Field(description="Utemeljitev za podano odločitev")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    is_history_related: str
    query: str
    relevance: str
    answer: str
    top_3: list[Document]
    relevant_part_texts: list[RelevantPassage]
    valid_rag_answer: str


retriever = TFIDFRetriever(k=10)
query_history_relation_parser = PydanticOutputParser(pydantic_object=QueryHistoryRelationParser)
query_classification_parser = PydanticOutputParser(pydantic_object=QueryClassificationParser)
top_3_parser = PydanticOutputParser(pydantic_object=Top3Response)
relevant_parts_parser = PydanticOutputParser(pydantic_object=RelevantPartsParser)
rag_answer_parser = PydanticOutputParser(pydantic_object=RAGAnswerParser)
answer_validation_parser = PydanticOutputParser(pydantic_object=AnswerValidationParser)


def serialize_chat_history(messages):
    serialized = ""
    for msg in messages:
        if msg.type == "human":
            serialized += f"Uporabnik: {msg.content}\n"
        elif msg.type == "ai":
            serialized += f"Pomočnik: {msg.content}\n"
    return serialized.strip()


def classify_query_relevance(state):
    print("-- Checking query relevance with AI Act --")

    chat_history = state["messages"][:-1]
    query = state["messages"][-1]  # HumanMessage(content="...")

    template = """
        Si pogovorni robot, specializiran za vprašanja o Evropskem aktu o umetni inteligenci (AI Act).

        Tvoja naloga:
        1. Analiziraj zadnji uporabnikov poziv.
        2. Če je potrebno, upoštevaj preteklo zgodovino pogovora za pravilno razumevanje konteksta.
        3. Razvrsti zadnje vprašanje v eno izmed dveh kategorij:
            - "AI Act" – če je vprašanje kakorkoli povezano z Evropskim zakonom/uredbo o umetni inteligenci.
            - "Not Related" – če vprašanje ni povezano z AI Act, ali je le splošen komentar/besedilo brez povezave.

        Vedno moraš vrniti veljaven JSON, obdan z blokom kode Markdown. Ne vračaj nobenega dodatnega besedila.
        NE ODGOVARJAJ NIČESAR DRUGEGA razen pravilnega JSON zapisa, v obliki kot je navedena spodaj v navodilih za strukturiranje odgovora.
        Če odgovor vsebuje karkoli drugega kot JSON, je NEVELJAVEN.

        ---

        Povzetek zakona o umetni inteligenci:
        {ai_act_summary}

        ---

        Zgodovina pogovora:
        {chat_history}

        ---

        Uporabnikov poziv:
        {query}

        ---

        Navodila za strukturiranje odgovora:
        {format_instructions}
    """

    prompt = PromptTemplate.from_template(template)

    # prompt = ChatPromptTemplate.from_messages([
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("system", """
    #         Si pogovorni robot, specializiran za vprašanja o Evropskem aktu o umetni inteligenci (AI Act).
    #
    #         Tvoja naloga:
    #         1. Analiziraj zadnje uporabnikovo vprašanje.
    #         2. Če je potrebno, upoštevaj zgornjo, preteklo zgodovino pogovora za pravilno razumevanje konteksta.
    #         3. Razvrsti zadnje vprašanje v eno izmed dveh kategorij:
    #             - "AI Act" – povezano z Evropsko uredbo o UI
    #             - "Not Related" – ni povezano
    #
    #         Vedno moraš vrniti veljaven JSON, obdan z blokom kode Markdown. Ne vračaj nobenega dodatnega besedila.
    #         NE ODGOVARJAJ NIČESAR DRUGEGA razen pravilnega JSON zapisa, v obliki kot je navedena spodaj v navodilih za strukturiranje odgovora.
    #         Če odgovor vsebuje karkoli drugega kot JSON, je NEVELJAVEN.
    #
    #         ---
    #
    #         Povzetek zakona o umetni inteligenci:
    #         {ai_act_summary}
    #
    #         ---
    #
    #         Navodila za strukturiranje odgovora:
    #         {format_instructions}
    #     """),
    #     query
    # ])

    chain = prompt | chat_model | query_classification_parser

    resp = chain.invoke({
        "query": query.content,
        "chat_history": serialize_chat_history(chat_history),
        "ai_act_summary": AI_ACT_SUMMARY,
        "format_instructions": query_classification_parser.get_format_instructions()
    })

    return {
        "relevance": resp.Relevance,
    }


def relevance_router(state):
    print("-- Router --")

    relevance = state["relevance"]
    if relevance == 'AI Act':
        print(">> DECISION: AI Act Related")
        return "RAG Call"
    elif relevance == 'Not Related':
        print(">> DECISION: Not AI Act Related")
        return "LLM Call"


def rephrase_query(state):
    print("-- Rephrasing user query into more suitable form for usage in RAG --")

    chat_history = state["messages"][:-1]
    query = state["messages"][-1]  # HumanMessage(content="...")

    template = """
        Tvoja naloga je:

        - Preoblikuj zadnji uporabnikov poziv v obliko, ki je čim bolj primerna za iskanje informacij v podatkovni bazi uredbe o umetni inteligenci. 
        - V pomoč ti je lahko spodnja zgodovina pogovora.
        - Odstrani vljudnostne fraze (npr. "živjo", "prosim", "hvala").
        - Če je poziv nejasen, ga naredi bolj specifičnega.
        - Če je poziv že jasen, ga pusti nespremenjenega.

        STROGA NAVODILA:
        - NE odgovarjaj na vprašanje uporabnika.
        - NE dodajaj nobenih dodatnih informacij ali razlag.
        - Samo vrni preoblikovano ali nespremenjeno besedilo poziva.
        - Odgovori samo z izboljšanim pozivom.

        Če ne upoštevaš teh pravil, je tvoj odgovor neveljaven.

        ---

        Zgodovina pogovora:
        {chat_history}

        ---

        Uporabnikov poziv:
        {query}

        ---

        Preoblikovan poziv:
        """

    prompt = PromptTemplate.from_template(template)

    # prompt = ChatPromptTemplate.from_messages([
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("system", """
    #         Tvoja naloga je:
    #
    #         - Preoblikuj zadnji uporabnikov poziv v obliko, ki je čim bolj primerna za iskanje informacij v podatkovni bazi.
    #         - Odstrani vljudnostne fraze (npr. "živjo", "prosim", "hvala").
    #         - Če je poziv nejasen, ga naredi bolj specifičnega.
    #         - Če je poziv že jasen, ga pusti nespremenjenega.
    #
    #         STROGA NAVODILA:
    #         - NE odgovarjaj na vprašanje uporabnika.
    #         - NE dodajaj nobenih dodatnih informacij ali razlag.
    #         - Samo vrni preoblikovano ali nespremenjeno besedilo poziva.
    #         - Odgovori samo z izboljšanim pozivom.
    #
    #         Če ne upoštevaš teh pravil, je tvoj odgovor neveljaven.
    #     """),
    #     query
    # ])

    chain = prompt | chat_model | StrOutputParser()

    resp = chain.invoke({
        "query": query.content,
        "chat_history": serialize_chat_history(chat_history),
    })

    print()
    print("REPHRASED QUERY: ", resp)
    print()

    return {
        "query": resp
    }


def rag_function(state):
    print("-- Calling RAG --")

    query = state["query"]

    template = """
    Spodaj so navedeni dokumenti, ki so bili pridobljeni na podlagi uporabnikovega vprašanja.

    Uporabnikovo vprašanje: {query}  
    
    Pridobljeni dokumenti (vsak ima ID in vsebino):
    <dokumenti>
    {context}
    </dokumenti>
    
    Tvoja naloga:
    1. Izmed vseh dokumentov izberi največ tri (3), ki so najbolj relevantni glede na vprašanje uporabnika.
    2. Vrni izključno **ID-je** teh dokumentov, kot so navedeni v vrstici "ID:" pri vsakem dokumentu.
    3. Ne vračaj vsebine dokumentov. Vrni samo ID-je.
    
    Pomembno:
    - Odgovori izključno v obliki pravilnega JSON-a, kot je določeno spodaj.
    - JSON mora biti ograjen z blokom markdown kode (začni z ```json in končaj z ```) — brez dodatnega besedila.
    - Če vrneš karkoli drugega kot JSON, je odgovor neveljaven.   
         
    Navodila za strukturiranje odgovora:
    {format_instructions}
    """

    retrieved_docs = retriever.invoke(query)

    # print(", ".join([
    #     f"ID: {doc.metadata['id']}"
    #     for doc in retrieved_docs
    # ]))

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    ).partial(format_instructions=top_3_parser.get_format_instructions())

    retrieval_chain = (
            RunnablePassthrough()
            | {
                "context": lambda _: get_context(retrieved_docs),
                "query": lambda x: x
            }
            | prompt
            | chat_model
            | top_3_parser
    )

    result = retrieval_chain.invoke(query)

    top_3 = [doc for doc in retrieved_docs if doc.metadata["id"] in result.DocumentIDs]

    # print()
    # print("TOP 3: ", top_3)
    # print()

    return {"top_3": top_3}


def get_context(retrieved_docs: list[Document]):
    context = "\n\n".join([
        f"ID: {doc.metadata['id']}\nVsebina:\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    return context


def llm_function(state):
    print("-- Calling LLM --")

    chat_history = state["messages"][:-1]
    query = state["messages"][-1]

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", """
            Odgovori na uporabnikovo vprašanje zgolj s svojim znanjem.
            Če nanj ne znaš odgovoriti, to odkrito povej.
        """),
        query
        # ("human", "{query}")
    ])

    chain = prompt | chat_model | StrOutputParser()

    response = chain.invoke({"chat_history": chat_history, "query": query})

    return {
        "answer": response,
        "relevant_part_texts": [],
        "messages": [AIMessage(content=response)]
    }


def rag_answer_function(state):
    print("-- Calling LLM For Answer From RAG --")

    query = state["query"]

    template = """
        Spodaj so podani dokumenti, ki naj bi bili najbolj relevantni glede na uporabnikovo vprašanje. Na njihovi podlagi oblikuj razumljiv odgovor na uporabnikovo vprašanje.
        
        **Pomembno:** 
        - Odgovora ne oblikuj na podlagi nobenih drugih virov ali zunanjega znanja. 
        - Uporabi zgolj znanje iz spodaj navedenih dokumentov in nikakor ne sklepaj na podlagi lastnih predpostavk ali informacij, ki niso v dokumentih.
        - Če dokumenti ne vsebujejo dovolj informacij za smiseln odgovor, vrni le prazen niz za polje "Answer".
        
        Poleg samega odgovora vrni tudi seznam najpomembnejših odlomkov iz dokumentov, ki so neposredno pripomogli k oblikovanju odgovora. Vsak odlomek mora biti:
        - **Dobesedno prepisan iz dokumenta**, brez sprememb, povzemanja ali sklepanja.
        - Označen z `id` dokumenta, iz katerega izvira.

        Uporabnikovo vprašanje: 
        <vprašanje>  
        {query}  
        </vprašanje>

        Relevantni dokumenti v pomoč pri tvorjenju odgovora:
        <dokumenti>
        {top_3}
        </dokumenti>

        Vedno moraš vrniti veljaven JSON, obdan z blokom kode Markdown. Ne vračaj nobenega dodatnega besedila.

        Navodila za strukturiranje odgovora:
        {format_instructions}
    """

    prompt = PromptTemplate(template=template, input_variables=["query", "top_3"]).partial(
        format_instructions=rag_answer_parser.get_format_instructions())

    chain = prompt | chat_model | rag_answer_parser

    response = chain.invoke({"query": query, "top_3": get_context(state["top_3"])})

    # print("ANSWER:")
    # print(response.Answer)

    # print("RELEVANT PARTS:")
    # for relevant_passage in response.RelevantParts:
    #     print(f"--- Document {relevant_passage.id} ---")
    #     print("Relevant Parts:", relevant_passage.text)
    # print()

    return {"answer": response.Answer, "relevant_part_texts": response.RelevantParts}


def validate_answer(state):
    print("-- Calling LLM To Check if RAG Answer Is Valid --")

    answer = state["answer"]
    query = state["query"]

    template = """
        Si pomočnik za preverjanje ustreznosti odgovorov.

        Tvoja naloga je, da preveriš, ali spodnji odgovor neposredno, popolno in smiselno odgovarja na uporabnikovo vprašanje.
        Odgovor mora biti vsebinsko povezan z vprašanjem in mora odgovoriti na tisto, kar uporabnik sprašuje — ne sme manjkati bistvenih informacij.
        Če je odgovor ustrezen, ga označi kot 'Valid'. Če ni, označi kot 'Invalid'.
        NE ODGOVARJAJ NIČESAR DRUGEGA razen pravilnega JSON zapisa, v obliki kot je navedena spodaj v navodilih za strukturiranje odgovora.

        Odgovor:
        \"\"\"{answer}\"\"\"

        Uporabnikovo vprašanje:
        \"\"\"{query}\"\"\"

        ---
        
        Vedno moraš vrniti veljaven JSON, obdan z blokom kode Markdown. Ne vračaj nobenega dodatnega besedila.

        Pričakovana struktura odgovora:
        {format_instructions}
        """

    prompt = PromptTemplate(
        input_variables=["answer", "query"],
        partial_variables={"format_instructions": answer_validation_parser.get_format_instructions()},
        template=template
    )

    chain = prompt | chat_model | answer_validation_parser

    response = chain.invoke({"answer": answer, "query": query})

    # print("\n", response, "\n")

    return {"valid_rag_answer": response.AnswerValid}


def invalid_rag_answer(state):
    print("-- Calling LLM For Invalid Answer --")

    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Razloži, da v svoji bazi znanja žal nisi uspel pridobiti dovolj informacij, za odgovor na zastavljeno vprašanje.
        """),
        ("human", "{query}")
    ])

    chain = prompt | chat_model | StrOutputParser()

    response = chain.invoke({"query": query})

    return {
        "answer": response,  # Overwrite RAG answer with new one
        "relevant_part_texts": [],  # We don't need this, since the answer obtained from those texts was invalid
        "messages": [AIMessage(content=response)]  # Append to chat history
    }


def valid_rag_answer(state):
    print("-- Appending Valid RAG Answer To Chat History --")

    valid_answer = state["answer"]

    return {
        "messages": [AIMessage(content=valid_answer)]  # Append valid RAG answer to chat history
    }


def answer_validation_router(state):
    print("-- RAG Answer Validation Router --")

    answer_valid = state["valid_rag_answer"]
    if answer_valid == 'Valid':
        print(">> DECISION: Valid")
        return "Valid"
    elif answer_valid == 'Invalid':
        print(">> DECISION: Invalid")
        return "Invalid"


def build_chatbot():
    workflow = StateGraph(AgentState)

    workflow.add_node("Classify_Query_Relevance", classify_query_relevance)
    workflow.add_node("Rephrase_Query", rephrase_query)
    workflow.add_node("RAG", rag_function)
    workflow.add_node("LLM", llm_function)
    workflow.add_node("RAG_Answer", rag_answer_function)
    workflow.add_node("Validate_RAG_Answer", validate_answer)
    workflow.add_node("Invalid_RAG_Answer", invalid_rag_answer)
    workflow.add_node("Valid_RAG_Answer", valid_rag_answer)

    workflow.set_entry_point("Classify_Query_Relevance")

    workflow.add_conditional_edges(
        "Classify_Query_Relevance",
        relevance_router,
        {
            "RAG Call": "Rephrase_Query",
            "LLM Call": "LLM",
        }
    )
    workflow.add_edge("Rephrase_Query", "RAG")
    workflow.add_edge("RAG", "RAG_Answer")
    workflow.add_edge("RAG_Answer", "Validate_RAG_Answer")
    workflow.add_conditional_edges(
        "Validate_RAG_Answer",
        answer_validation_router,
        {
            "Valid": "Valid_RAG_Answer",
            "Invalid": "Invalid_RAG_Answer"
        }
    )
    workflow.add_edge("LLM", END)
    workflow.add_edge("Invalid_RAG_Answer", END)
    workflow.add_edge("Valid_RAG_Answer", END)

    # memory = MemorySaver() # Chat history persists only on current script run
    memory = SqliteSaver(sqlite_conn)  # Chat history persists over multiple runs

    chatbot = workflow.compile(checkpointer=memory)

    return chatbot
