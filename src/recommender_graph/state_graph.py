from typing import List, TypedDict

class State(TypedDict):

    query: str
    on_topic: bool
    recommendation: str
    products: str
    self_query_state: str
