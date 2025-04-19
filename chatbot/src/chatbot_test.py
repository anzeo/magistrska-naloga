import dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from chatbot.src.TFIDFRetriever import TFIDFRetriever

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, verbose=True)
retriever = TFIDFRetriever(k=10)

memory = ConversationBufferMemory(return_messages=True, memory_key="history")


def format_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, HumanMessage):
            buffer += "\nHuman Question: " + dialogue_turn.content
        elif isinstance(dialogue_turn, AIMessage):
            buffer += "\nChatbot Answer: " + dialogue_turn.content
        elif isinstance(dialogue_turn, SystemMessage):
            buffer += "\nSystem: " + dialogue_turn.content
    return buffer


relevance_check_chain_prompt_template = """
Si pogovorni robot, specializiran za odgovarjanje na vprašanja o Evropskem zakonu o umetni inteligenci (AI Act).
Tvoja naloga je oceniti, ali je vprašanje povezano z zakonom o umetni inteligenci, z zgodovino pogovora ali z nobenim od njiju.  
Pri odločanju si lahko pomagaš z opisom zakona in zgodovino pogovora, ki sta podana spodaj.
Odgovori izključno z eno od treh možnosti: "ai_act", "zgodovina", "nepovezano".  

---

Kontekst za lažje razumevanje zakona o umetni inteligenci:
- Prvi pravni okvir za regulacijo umetne inteligence v EU.
- Cilj: Zagotoviti varno in etično uporabo umetne inteligence ter zaščititi temeljne pravice državljanov.  
- Razvrstitev sistemov umetne inteligence:
  - Nesprejemljivo tveganje: Prepovedani sistemi (socialno točkovanje, manipulativna umetna inteligenca).  
  - Visoko tveganje: Sistemi v zdravstvu, izobraževanju, varnosti (zahtevana stroga regulacija).
  - Omejeno tveganje: Sistemi, pri katerih morajo biti uporabniki obveščeni o njihovi uporabi.
  - Minimalno tveganje: Večina sistemov, kot so priporočilni algoritmi, ni strogo regulirana.
- Evropski odbor za umetno inteligenco nadzira izvajanje zakona.
---

Zgodovina pogovora:
{history}

---

Uporabnikovo vprašanje:
{input}

---

Odgovor:
"""

irrelevant_input_chain_prompt_template = """
Najprej poskusi odgovoriti na vprašanje čim bolj jasno in natančno, če ga razumeš in lahko podaš ustrezen odgovor.
Pri snovanju odgovora si lahko pomagaš z zgodovino pogovora, ki je podana spodaj.
Če odgovora ne poznaš, to odkrito povej, vendar se vedno najprej potrudi najti smiseln odgovor.
Le, če res ne moreš odgovoriti, uporabniku povej, kje lahko najde ustrezne informacije.

---

Uporabnikovo vprašanje:
<vprašanje>
{input}
</vprašanje>  

Zgodovina pogovora:  
<zgodovina>
{history}
</zgodovina>  

Tvoj odgovor:
"""

relevant_input_chain_prompt_template = """
Spodaj so navedeni dokumenti, ki so bili pridobljeni na podlagi uporabnikovega vprašanja.

Uporabnikovo vprašanje: 
<vprašanje>  
{input}  
</vprašanje>

Pridobljeni dokumenti:
<dokumenti>
{context}
</dokumenti>

Zgodovina pogovora:
<zgodovina>
{history}
</zgodovina>

Prosim, izberi tri (3) najbolj relevantne dokumente glede na uporabnikovo poizvedbo. 
Vrni samo njihove izvirne besedilne vsebine, popolnoma nespremenjene. Besedilo mora ostati natanko takšno, kot je podano.
Lahko se zgodi, da ti ne bo uspelo najti treh dovolj relevantnih dokumentov. V tem primeru jih lahko vrneš manj.

Odgovor naj bo v sledečem formatu: ["vsebina_1", "vsebina_2", "vsebina_3"]
Pri tem je "vsebina_n" nespremenjeno besedilo dokumenta.
"""

form_answer_prompt_template = """
Spodaj so podani dokumenti, ki naj bi bili najbolj relevantni glede na uporabnikovo vprašanje. Na njihovi podlagi oblikuj razumljiv odgovor na uporabnikovo vprašanje.
Ni nujno, da so vsi dokumenti dovolj relevantni za tvorjenje odgovora, zato uporabi le tiste, ki so ti v pomoč. Pri tvorjenju uporabljaj zgolj znanje iz navedenih dokumentov.
Če iz dokumentov ne znaš ustvariti ustreznega odgovora na vprašanje, vrni le prazen niz.

Uporabnikovo vprašanje: 
<vprašanje>  
{input}  
</vprašanje>

Relevantni dokumenti v pomoč pri tvorjenju odgovora:
<dokumenti>
{top_3}
</dokumenti>

Odgovor:
"""

form_answer_from_history_prompt_template = """
Spodaj je podano uporabnikovo vprašanje in zgodovina pogovora. Na podlagi tega oblikuj jasen in razumljiv odgovor na uporabnikovo vprašanje.
Če iz zgodovine pogovora lahko oblikuješ odgovor na vprašanje o tem, kaj je uporabnik vprašal prej, to jasno povej in povzetek prejšnjega vprašanja vrni. Če zgodovina ne vsebuje nobenega odgovora na vprašanje, to odkrito povej.

Uporabnikovo vprašanje:
<vprašanje>
{input}
</vprašanje>

Zgodovina pogovora:
<zgodovina>
{history}
</zgodovina>

Odgovor:
"""

answer_quality_check_chain_prompt_template = """
Spodaj sta podana uporabnikovo vprašanje in odgovor, ki poskuša nanj odgovoriti.
Tvoja naloga je oceniti, ali je odgovor relevanten glede na zastavljeno vprašanje in ali nanj ustrezno odgovori.
Odgovori izključno z "da" ali "ne" in ne dodajaj dodatnih razlag. Če odgovor ne zadošča zastavljenemu vprašanju odgovori z "ne", sicer z "da"  

Uporabnikovo vprašanje:  
<vprašanje>  
{input}  
</vprašanje>

Možen odgovor na vprašanje:
<odgovor_na_vprasanje>
{answer}
</odgovor_na_vprasanje>

Zgodovina pogovora:
<zgodovina>
{history}
</zgodovina>

Odgovor:
"""

bad_answer_chain_prompt_template = """
Si pogovorni robot, specializiran za odgovarjanje na vprašanja o Evropskem zakonu o umetni inteligenci (AI Act).  
Podaj razlago, da na zastavljeno vprašanje žal ne znaš odgovoriti, saj v svoji bazi znanja nimaš dovolj informacij, o Evropskem zakonu o umetni inteligenci (AI Act).

Uporabnikovo vprašanje:  
<vprašanje>  
{input}  
</vprašanje>

Zgodovina pogovora:
<zgodovina>
{history}
</zgodovina>

Odgovor:
"""


def route(relevance):
    if "ai_act" in relevance["relevance"].lower():
        print("Vprašanje JE relevantno")
        # return relevant_input_chain
    elif "zgodovina" in relevance["relevance"].lower():
        print("Vprašanje JE povezano z zgodovino pogovora")
        # return relevant_input_chain
    elif "nepovezano" in relevance["relevance"].lower():
        print("Vprašanje NI relevantno")
        return irrelevant_input_chain


def route_answer_quality(data):
    if "da" in data["answer_quality"].lower():
        print("Odgovor JE ustrezen")
        return data["answer"]
    elif "ne" in data["answer_quality"].lower():
        print("Odgovor NI ustrezen")
        return (
                PromptTemplate.from_template(bad_answer_chain_prompt_template)
                | chat_model
                | StrOutputParser())


relevance_check_chain = RunnablePassthrough.assign(
    relevance=(
            PromptTemplate.from_template(relevance_check_chain_prompt_template)
            | chat_model
            | StrOutputParser()
    )
)

irrelevant_input_chain = RunnablePassthrough.assign(
    answer=(
            PromptTemplate.from_template(irrelevant_input_chain_prompt_template)
            | chat_model
            | StrOutputParser()
    )
)


def retrieve_docs(input_data):
    return {**input_data, "documents": retriever.invoke(input_data["input"])}


def get_context(retrieved_docs):
    context = "\n\n".join(
        [f"<dokument>\n{doc.page_content}\n</dokument>" for i, doc in
         enumerate(retrieved_docs["documents"])],
    )
    return {**retrieved_docs, "context": context}


def top_3(input_data):
    return """
    <dokument>
    KONČNE DOLOČBE
    Začetek veljavnosti in uporaba
    Ta uredba začne veljati dvajseti dan po objavi vUradnem listu Evropske unije.
    Uporablja se od 2. avgusta 2026.
    Vendar se:
    (a) poglavji I in II začneta uporabljati od 2. februarja 2025;
    (b) poglavje III, oddelek 4, poglavje V, poglavje VII in poglavje XII ter člen 78 uporabljajo od 2. avgusta 2025, razen člena 101;
    (c) člen 6(1) in ustrezne obveznosti iz te uredbe uporabljajo od 2. avgusta 2027.
    </dokument>
    
    <dokument>
    Ta uredba se uporablja od 2. avgusta 2026. Vendar bi bilo treba zaradi nesprejemljivega tveganja, povezanega z uporabo UI na določene načine, prepovedi ter splošne določbe te uredbe uporabljati že od 2. februarja 2025. Čeprav bodo te prepovedi polno učinkovale šele po vzpostavitvi upravljanja in izvrševanja te uredbe, je pomembno predvideti uporabo prepovedi, da se upoštevajo nesprejemljiva tveganja in zato, da bi to vplivalo na druge postopke, kot denimo v civilnem pravu. Poleg tega bi morala infrastruktura, povezana z upravljanjem in sistemom ugotavljanja skladnosti, začeti delovati pred 2. avgustom 2026, zato bi bilo treba določbe o priglašenih organih in strukturi upravljanja uporabljati od 2. avgusta 2025. Glede na hiter tehnološki napredek in sprejetje modelov UI za splošne namene bi bilo treba obveznosti za ponudnike modelov UI za splošne namene uporabljati od 2. avgusta 2025. Kodekse prakse bi bilo treba pripraviti najpozneje do 2. maja 2025, da bi ponudniki lahko pravočasno dokazali skladnost. Urad za UI bi moral zagotavljati posodabljanje pravil in postopkov razvrščanja glede na tehnološki razvoj. Poleg tega bi morale države članice določiti pravila o kaznih, vključno z upravnimi globami, in o njih uradno obvestiti Komisijo ter do datuma začetka uporabe te uredbe zagotoviti, da jih bodo učinkovito izvajale. Zato bi bilo treba določbe o kaznih uporabljati od 2. avgusta 2025.
    </dokument>
    
    <dokument>
    Visokotvegani sistemi UI bi morali biti zasnovani in razviti tako, da lahko fizične osebe nadzorujejo njihovo delovanje, zagotavljajo, da se uporabljajo, kot je bilo predvideno, in da se njihovi vplivi obravnavajo v celotnem življenjskem ciklu sistema. V ta namen bi moral ponudnik sistema pred dajanjem sistema na trg ali v uporabo določiti ustrezne ukrepe za človeški nadzor. Zlasti bi morali taki ukrepi, kadar je primerno, zagotavljati, da za sistem veljajo vgrajene operativne omejitve, ki jih sistem sam ne more razveljaviti, da se sistem odziva na človeškega operaterja ter da imajo fizične osebe, ki jim je bil dodeljen človekov nadzor, potrebne kompetence, usposobljenost in pooblastila za opravljanje te vloge. Prav tako je bistveno po potrebi zagotoviti, da visokotvegani sistemi UI vključujejo mehanizme za usmerjanje in obveščanje fizične osebe, ki ji je bil dodeljen človeški nadzor, za sprejemanje informiranih odločitev o tem, ali, kdaj in kako posredovati, da bi se izognili negativnim posledicam ali tveganjem, ali ustaviti sistem, če ne deluje, kot je bilo predvideno. Glede na pomembne posledice, ki jih napačno ujemanje v nekaterih sistemih za biometrično identifikacijo ljudi pomeni za osebe, je za te sisteme ustrezno določiti okrepljeno zahtevo po človekovem nadzoru, tako da uvajalec ne more sprejeti ukrepa ali odločitve na podlagi identifikacije, ki izhaja iz sistema, če tega nista ločeno preverili in potrdili vsaj dve fizični osebi. Ti osebi sta lahko iz enega ali več subjektov in vključujeta osebo, ki upravlja ali uporablja sistem. Ta zahteva ne bi smela ustvariti nepotrebnega bremena ali zamud, in lahko bi zadoščalo, da bi se ločena preverjanja različnih oseb samodejno zabeležila v dnevnikih, ki jih ustvari sistem. Glede na posebnosti področij preprečevanja, odkrivanja in preiskovanja kaznivih dejanj, migracij, nadzora meje in azila se ta zahteva ne bi smela uporabljati, kadar je v skladu s pravom Unije ali nacionalnim pravom uporaba te zahteve nesorazmerna.
    </dokument>
    """


top_3_sub_chain = (
        RunnableLambda(retrieve_docs)
        | RunnableLambda(get_context)
        | PromptTemplate.from_template(relevant_input_chain_prompt_template)
        | chat_model
        | StrOutputParser()
)

form_answer_sub_chain = (
        PromptTemplate.from_template(form_answer_prompt_template)
        | chat_model
        | StrOutputParser()
)

form_answer_from_history_sub_chain = (
        PromptTemplate.from_template(form_answer_from_history_prompt_template)
        | chat_model
        | StrOutputParser()
)

answer_quality_check_chain = (
        PromptTemplate.from_template(answer_quality_check_chain_prompt_template)
        | chat_model
        | StrOutputParser()
)

relevant_input_chain = (
        RunnablePassthrough.assign(top_3=top_3_sub_chain)
        | RunnablePassthrough.assign(answer=form_answer_sub_chain)
        | RunnablePassthrough.assign(answer_quality=answer_quality_check_chain)
)

chat_history_relevant_input_chain = (
    RunnablePassthrough.assign(answer=form_answer_from_history_sub_chain)
)

if __name__ == "__main__":
    while True:

        user_input = input("Your message: ")
        # print(full_chain.invoke({"input": "Ali obstajajo orodja umetne inteligence, ki so v šolah prepovedana, ker bi lahko manipulirala z učenci ali posegala v njihovo zasebnost?"}))

        formated_chat_history = format_chat_history(memory.load_memory_variables({})["history"])

        resp = relevance_check_chain.invoke(
            {
                "input": user_input,
                "history": formated_chat_history
            }
        )

        if "ai_act" in resp["relevance"].lower():
            print("Vprašanje JE relevantno")
            resp = relevant_input_chain.invoke(
                {
                    "input": user_input,
                    "history": formated_chat_history
                }
            )
        elif "zgodovina" in resp["relevance"].lower():
            print("Vprašanje JE povezano z zgodovino pogovora")
            resp = chat_history_relevant_input_chain.invoke(
                {
                    "input": user_input,
                    "history": formated_chat_history
                }
            )
        elif "nepovezano" in resp["relevance"].lower():
            print("Vprašanje NI relevantno")
            resp = irrelevant_input_chain.invoke(
                {
                    "input": user_input,
                    "history": formated_chat_history
                }
            )

        print(resp["answer"])
        print()

        memory.save_context({"input": user_input}, {"answer": resp["answer"]})
