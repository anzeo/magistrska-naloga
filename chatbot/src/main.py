from typing import TypedDict, Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel, Field

from TFIDFRetriever import TFIDFRetriever
from ai_act_summary import AI_ACT_SUMMARY

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


def is_query_history_related(state):
    print("-- Checking if query is related to chat history --")

    chat_history = state["messages"]
    query = state["messages"][-1]  # HumanMessage(content="...")

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", """
            You are an assistant that classifies whether the latest user message is related to previous messages in the conversation.

            A message is considered "History Related" if:
            - It logically follows from, continues, or is on the same topic as any earlier user messages.
            - It refers explicitly or implicitly (e.g., posing a follow-up question) to a topic already introduced earlier in the conversation.
        
            If the latest message introduces a new, unrelated topic, respond with "Not History Related".
            Respond ONLY with "History Related" or "Not History Related".
            
            You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
            
            {format_instructions}
        """),
    ])

    chain = prompt | chat_model | query_history_relation_parser

    response = chain.invoke({
        "chat_history": chat_history,
        "format_instructions": query_history_relation_parser.get_format_instructions()
    })

    # print(prompt.invoke({
    #     "chat_history": chat_history,
    # }))

    state_result = {"is_history_related": response.Related}

    if response.Related == 'Not History Related':
        # Set query as it is. If it's history related, the reformated query will be set in create_history_aware_query()
        state_result["query"] = query.content

    return state_result


def is_query_history_related_router(state):
    print("-- Router --")

    relevance = state["is_history_related"]
    if relevance == 'History Related':
        print(">> DECISION: History Related")
        return "History Related"
    elif relevance == 'Not History Related':
        print(">> DECISION: Not History Related")
        return "Not History Related"


def create_history_aware_query(state):
    print("-- Reformating user query into standalone question --")

    chat_history = state["messages"][:-1]
    query = state["messages"][-1]  # HumanMessage(content="...")

    # REFERENCE QUERY
    # "Given a chat history and the latest user question "
    # "which might reference context in the chat history, "
    # "formulate a standalone question which can be understood "
    # "without the chat history. Do NOT answer the question, "
    # "just reformulate it if needed and otherwise return it as is."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Si pomočnik, katerega edina naloga je preoblikovanje uporabniških vnosov v samostojna, jasno razumljiva vprašanja ali izjave, ki jih je mogoče razumeti brez zgodovine pogovora. Ne odgovarjaš na vprašanja in ne dodajaš nobenih dodatnih informacij, razlag, vljudnostnih fraz ali napovedi. Upoštevaj spodnja pravila.
            Glede na zgodovino pogovora in uporabnikov vnos, ki se lahko nanaša na kontekst iz zgodovine pogovora, oblikuj samostojno poizvedbo, ki jo je mogoče razumeti brez zgodovine pogovora.

            **Zelo pomembno**:
            - Tvoja naloga je IZKLJUČNO preoblikovanje uporabnikovega vnosa v samostojen vnos, če je to potrebno.
            - NIKOLI ne dodajaj nobenih novih informacij, odgovorov, razlag, napovedi, domnev ali zaključkov.
            - NIKOLI ne vstavljaj nobenih vsebin, ki niso bile eksplicitno podane v uporabniškem vnosu ali zgodovini pogovora.
            - Če uporabnikov vnos že vsebuje vse potrebno za samostojno razumevanje, ga vrni v nespremenjeni obliki.
            - Če uporabnikov vnos vsebuje navodila o obliki odgovora (npr. 'na kratko', 'v alinejah', 'v obliki povzetka', 'podrobno', 'v enem stavku' itd.), jih MORAŠ ohraniti popolnoma enaka v preoblikovanem vnosu.
            - Če uporabniški vnos ni vprašanje (npr. pozdrav, komentar, ali splošna izjava brez vprašanja), ga vrni v nespremenjeni obliki.
            - Če je uporabnikov vnos vprašanje, ki potrebuje kontekst iz zgodovine pogovora, uporabi relevantne dele zgodovine in oblikuj samostojno vprašanje.
            - Če je vprašanje že samostojno, ga vrni v nespremenjeni obliki.

            **Dodatno**:
            - Ne poskušaj ugibati, napovedovati ali domnevati odgovora na vprašanje.
            - Ne vstavljaj nobenih pojasnil, odgovorov, vljudnostnih fraz, uvodnih stavkov ali sklepov.

            **Primeri**:
            - Uporabnik: "Živjo" → vrni: "Živjo"
            - Uporabnik: "Kaj določa uredba o umetni inteligenci?" → vrni nespremenjeno
            - Uporabnik: "Kaj pa glede varnosti?" (če je bil prej pogovor o umetni inteligenci) → preoblikuj v: "Kaj določa uredba o umetni inteligenci glede varnosti?"
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        query
    ])

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", """
    #         Si pomočnik, katerega edina naloga je preoblikovanje uporabniških vnosov v samostojna, jasno razumljiva vprašanja ali izjave, ki jih je mogoče razumeti brez zgodovine pogovora. Ne odgovarjaš na vprašanja in ne dodajaš nobenih dodatnih informacij, razlag, vljudnostnih fraz ali napovedi. Upoštevaj spodnja pravila.
    #         Glede na zgodovino pogovora in uporabnikov vnos, ki se lahko nanaša na kontekst iz zgodovine pogovora, oblikuj samostojno poizvedbo, ki jo je mogoče razumeti brez zgodovine pogovora.
    #
    #         **Zelo pomembno**:
    #         - Tvoja naloga je IZKLJUČNO preoblikovanje uporabnikovega vnosa v samostojen vnos, če je to potrebno.
    #         - NIKOLI ne dodajaj nobenih novih informacij, odgovorov, razlag, napovedi, domnev ali zaključkov.
    #         - NIKOLI ne vstavljaj nobenih vsebin, ki niso bile eksplicitno podane v uporabniškem vnosu ali zgodovini pogovora.
    #     """),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     query
    # ])

    #   Ne odgovarjaj na vprašanje — če je potrebno, ga samo preoblikuj, sicer pa ga vrni v nespremenjeni obliki.
    #   Pomembno:
    #   Če uporabnikovo vprašanje vključuje kakršnakoli navodila o obliki odgovora (npr. 'na kratko', 'v alinejah', 'v obliki povzetka', 'podrobno', itd.), jih OBVEZNO ohrani v preoblikovanem vprašanju na isti način kot so zapisani v izvirnem vprašanju.

    chain = prompt | chat_model | StrOutputParser()

    resp = chain.invoke({
        "chat_history": chat_history,
    })

    print("\nREFORMATED QUERY: ", resp, "\n")

    return {"query": resp}


def classify_query_relevance(state):
    print("-- Checking query relevance with AI Act --")

    # chat_history = state["messages"][:-1]
    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        # MessagesPlaceholder(variable_name="chat_history"),
        ("system", """
            Si pogovorni robot, specializiran za vprašanja o Evropskem aktu o umetni inteligenci (AI Act).
            Tvoja naloga je, da uporabnikovo poizvedbo razvrstiš v eno izmed naslednjih dveh kategorij: [AI Act, Not Related].
            Pri odločanju se opiraj na spodnji opis zakona o umetni inteligenci (AI Act).
            Odgovori izključno z eno izmed dveh možnosti:
              - "AI Act" – če je poizvedba kakorkoli povezana z Evropskim zakonom/uredbo o umetni inteligenci
              - "Not Related" – če poizvedba ni povezana z Evropskim zakonom/uredbo o umetni inteligenci, ali če je zgolj splošen komentar ali katerikoli drug tekst brez povezave z uredbo.
            Vedno moraš vrniti veljaven JSON, obdan z blokom kode Markdown. Ne vračaj nobenega dodatnega besedila.
            NE ODGOVARJAJ NIČESAR DRUGEGA razen pravilnega JSON zapisa, v obliki kot je navedena spodaj v navodilih za strukturiranje odgovora.
            Če odgovor vsebuje karkoli drugega kot JSON, je NEVELJAVEN.

            ---
            Kontekst za lažje razumevanje zakona o umetni inteligenci:
            {ai_act_summary}
            
            ---
            
            Navodila za strukturiranje odgovora:
            {format_instructions}
        """),
        ("human", "{query}")
    ])

    chain = prompt | chat_model | query_classification_parser

    resp = chain.invoke({
        # "chat_history": chat_history,
        "query": query,
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

    # print("TOP 3:")
    # print(top_3)
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
    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", """
            Odgovori na uporabnikovo vprašanje zgolj s svojim znanjem.
            Če nanj ne znaš odgovoriti, to odkrito povej.
        """),
        ("human", "{query}")
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


def relevant_parts_function(state):
    print("-- Calling LLM To Mark Relevant Parts Of Documents --")

    answer = state["answer"]
    top_3 = state["top_3"]

    template = """
        Si pomočnik, ki preverja, ali je spodnji odgovor podprt z vsebino dokumenta.

        Tvoja naloga:
        1. Preveri, ali je spodnji odgovor podprt s tekstom iz dokumenta. Odgovori izključno z eno izmed možnosti: `"Da"` ali `"Ne"`.
        2. Če je odgovor "Da", iz dokumenta **dobesedno prekopiraj vse najbolj relevantne odlomke**, ki potrjujejo odgovor.
            Pomembno:
            - Odlomki morajo biti **dobesedno prepisani iz dokumenta**, brez sprememb, povzemanja ali sklepanja.
            - **Ne vključuj besedila odgovora v relevantne odlomke**. Odlomki morajo izvirati iz dokumenta, ne pa iz odgovora.
        3. Če odgovor ni podprt z vsebino dokumenta, vrni prazen seznam `[]` za relevantne odlomke.
        4. Če odgovor ni 100 % podprt s tekstom dokumenta, moraš vrniti `"Ne"`.
        
        ---
    
        Odgovor (ki ga preverjaš):
        \"\"\"{answer}\"\"\"
    
        Vsebina dokumenta (edini vir resnice):
        \"\"\"{document}\"\"\"
    
        ---
    
        Navodila:
        - Odgovori v obliki veljavnega JSON-objekta po spodnjem formatu.
        - JSON mora biti ograjen z blokom kode v markdown obliki (začni z ```json in končaj z ```).
        - Ne dodaj nobenih dodatnih razlag ali besedila izven JSON-a.
    
        Pričakovana struktura odgovora:
        {format_instructions}
        """

    prompt = PromptTemplate(
        input_variables=["answer", "document"],
        partial_variables={"format_instructions": relevant_parts_parser.get_format_instructions()},
        template=template
    )

    chain = prompt | chat_model | relevant_parts_parser

    print()
    relevant_part_texts = []
    for i, document in enumerate(top_3):
        response = chain.invoke({"answer": answer, "document": document.page_content})
        print(f"--- Document {document.metadata['id']} ---")
        print("Included:", response.AnswerIncluded)
        print("Relevant Parts:", response.RelevantParts)
    print()

    return {"relevant_part_texts": relevant_part_texts}


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


if __name__ == "__main__":
    workflow = StateGraph(AgentState)

    workflow.add_node("Is_Related_to_History", is_query_history_related)
    workflow.add_node("History_Aware_Query", create_history_aware_query)
    workflow.add_node("Classify_Query_Relevance", classify_query_relevance)
    workflow.add_node("RAG", rag_function)
    workflow.add_node("LLM", llm_function)
    workflow.add_node("RAG_Answer", rag_answer_function)
    workflow.add_node("Validate_RAG_Answer", validate_answer)
    workflow.add_node("Invalid_RAG_Answer", invalid_rag_answer)
    workflow.add_node("Valid_RAG_Answer", valid_rag_answer)

    # workflow.set_entry_point("Is_Related_to_History")
    workflow.set_entry_point("History_Aware_Query")

    # workflow.add_conditional_edges(
    #     "Is_Related_to_History",
    #     is_query_history_related_router,
    #     {
    #         "History Related": "History_Aware_Query",
    #         "Not History Related": "Classify_Query_Relevance",
    #     }
    # )
    workflow.add_edge("History_Aware_Query", "Classify_Query_Relevance")
    workflow.add_conditional_edges(
        "Classify_Query_Relevance",
        relevance_router,
        {
            "RAG Call": "RAG",
            "LLM Call": "LLM",
        }
    )
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

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    chatbot = workflow.compile(checkpointer=memory)

    try:
        png_data = chatbot.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("Image saved as graph.png")
    except Exception as e:
        print(f"Error: {e}")

    while True:
        user_input = input("Your message: ")

        # Reset everything, except messages, where chat history is stored
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "query": None,
            "relevance": None,
            "answer": None,
            "top_3": [],
            "relevant_part_texts": [],
            "valid_rag_answer": None
        }

        config = {"configurable": {"thread_id": "2"}}
        chatbot_response = chatbot.invoke(inputs, config)

        print("\nANSWER: ", chatbot_response["answer"], "\n")

        relevant_part_texts = chatbot_response.get("relevant_part_texts", [])
        if relevant_part_texts:
            print("RELEVANT PARTS:")
            for relevant_passage in chatbot_response.get("relevant_part_texts", []):
                print(f"--- Document {relevant_passage.id} ---")
                print("Relevant Parts:", relevant_passage.text)
            print()

        # for event in chatbot.stream(inputs, config, stream_mode="values"):
        #     event["messages"][-1].pretty_print()
