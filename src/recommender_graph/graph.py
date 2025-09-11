import sys
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.recommender_graph.topic_check import topic_classifier
from src.recommender_graph.rag import rag_recommender
from src.recommender_graph.ranker import ranker_node
from src.recommender_graph.self_query import self_query_retriever
from src.recommender_graph.state_graph import State




def create_recommender_graph():

    workflow = StateGraph(State)

    workflow.add_node("self_query_retrieve", self_query_retriever)
    workflow.add_node("rag_recommender", rag_recommender)
    workflow.add_node("ranker", ranker_node)
    workflow.add_node("check_topic", topic_classifier)

    workflow.add_edge("ranker", "rag_recommender")
    workflow.add_edge("rag_recommender", END)

    workflow.set_entry_point("check_topic")
    workflow.add_conditional_edges(
        "check_topic", 
        lambda state: state["on_topic"],
        {"Yes": "self_query_retrieve", "No": END}
    )

    workflow.add_conditional_edges(
        "self_query_retrieve",
        lambda state: state["self_query_state"],
        {"success": "rag_recommender", "empty": "ranker"}
    )

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


if __name__ == "__main__": 
    app = create_recommender_graph()
    config = {"configurable": {"thread_id": "1"}}
    state = {"query": "A long black shirt"}
    output = app.invoke(state, config=config)

    print(output)