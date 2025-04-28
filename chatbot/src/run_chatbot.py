"""
Script that runs chatbot in terminal
"""
from langchain_core.messages import HumanMessage
from src.core.chatbot import build_chatbot, MemoryType

chatbot = build_chatbot(MemoryType.MEMORY)

# try:
#     png_data = chatbot.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
#     with open("graph.png", "wb") as f:
#         f.write(png_data)
#     print("Image saved as graph.png")
# except Exception as e:
#     print(f"Error: {e}")

if __name__ == '__main__':
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

        config = {"configurable": {"thread_id": "1"}}
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
