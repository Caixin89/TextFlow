def load_messages(prompt, image=None):
    # Text only messages (Note that VLMs can be used as LLMs without image input)
    if not image:
        return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
    return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

def load_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_number_of_nodes",
                "description": "Returns the number of nodes in the flowchart.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_number_of_edges",
                "description": "Returns the number of edges in the flowchart.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_successors",
                "description": "Returns the direct successors (outgoing connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_predecessors",
                "description": "Returns the direct predecessors (incoming connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shortest_path_length",
                "description": "Returns the number of edges in the shortest path between two nodes based on their descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_node_description": {
                            "type": "string",
                            "description": "The description of the start node.",
                        },
                        "end_node_description": {
                            "type": "string",
                            "description": "The description of the end node.",
                        },
                    },
                    "required": ["start_node_description", "end_node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_indegree",
                "description": "Returns the maximum indegree (number of incoming edges) for any node.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },                
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_outdegree",
                "description": "Returns the maximum outdegree (number of outgoing edges) for any node.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },                
            },
        },
    ]
    return tools


def load_tools_code():
    return """
def get_number_of_nodes():
    return flowchart.get_number_of_nodes()

def get_number_of_edges():
    return flowchart.get_number_of_edges()

def get_direct_successors(node_description):
    return flowchart.get_direct_successors(node_description)

def get_direct_predecessors(node_description):
    return flowchart.get_direct_predecessors(node_description)

def get_shortest_path_length(node1, node2):
    return flowchart.get_shortest_path_length(node1, node2)

def get_max_indegree():
    return flowchart.get_max_indegree()

def get_max_outdegree():
    return flowchart.get_max_outdegree()
"""
