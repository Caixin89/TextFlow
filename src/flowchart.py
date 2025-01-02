import json
import re
from collections import deque


class Mermaid2Flowchart:
    def __init__(self, mermaid_code):
        self.mermaid_code = mermaid_code
        self.flowchart_code = (
            "from flowchart import Flowchart\n\nflowchart = Flowchart()\n"
        )
        self.nodes = set()  # Use a set to avoid duplicates
        self.edges = []

    def simplify_mermaid_code(self):
        # Remove content inside parentheses, square brackets, and curly braces
        return re.sub(r"(\[.*?\]|\(.*?\)|\{.*?\})", "", self.mermaid_code)

    def parse_mermaid_code(self):
        # Separate nodes and edges from the mermaid code
        lines = self.mermaid_code.strip().split("\n")

        for line in lines:
            if "-->" in line:  # Edge definition
                # Extract node definitions from the edge lines
                parts = re.split(
                    r"-->|(\|.*\|)", line
                )  # Split by --> and any labels like |Yes|
                for part in parts:
                    if part and not part.startswith(
                        "|"
                    ):  # Avoid labels and empty parts
                        node = part.strip()
                        if node:
                            self.nodes.add(node)
            else:  # Node definition only
                self.nodes.add(line.strip())

        # Simplify the Mermaid code by removing node content
        simplified_code = self.simplify_mermaid_code()

        # Separate nodes and edges from the simplified code
        lines = simplified_code.strip().split("\n")

        for line in lines:
            if "-->" in line:  # Edge definition
                self.edges.append(line.strip())

    def generate_nodes(self):
        # Updated regex to handle different node formats based on the shape conventions:
        # Ellipses for ([ ]), rectangles for [ ], parallelograms for [/ /], and diamonds for { }
        node_pattern = re.compile(
            r"(\w+)\(\[\s?(.*?)\s?\]\)"  # Ellipses: A([Start])
            r"|(\w+)\[\/\s?(.*?)\s?\/\]"  # Parallelograms: A[/Some content/]
            r"|(\w+)\[\s?(.*?)\s?\]"  # Rectangles: A[Some content]
            r"|(\w+)\{\s?(.*?)\s?\}"  # Diamonds: A{Some decision}
        )

        node_lines = []
        for node in self.nodes:
            match = node_pattern.match(node)
            if match:
                if match.group(1):  # Ellipse (start/end), e.g., A([Start])
                    node_id = match.group(1)
                    content = match.group(2)
                    shape = "ellipse"
                elif match.group(3):  # Parallelogram, e.g., A[/Some content/]
                    node_id = match.group(3)
                    content = match.group(4)
                    shape = "parallelogram"
                elif match.group(5):  # Rectangle, e.g., A[Some content]
                    node_id = match.group(5)
                    content = match.group(6)
                    shape = "box"
                elif match.group(7):  # Diamond (decision), e.g., A{Some decision}
                    node_id = match.group(7)
                    content = match.group(8)
                    shape = "diamond"

                # Clean up the content by removing unnecessary quotes or spaces
                if content:
                    content = content.replace('"', "").strip()
                else:
                    content = ""

                # Add the node line to the list, but don't append it to the code yet
                node_lines.append(
                    f'flowchart.add_node("{node_id}", "{content}", "{shape}")\n'
                )

        # Sort the node lines alphabetically by the node ID
        node_lines.sort()

        # Append the sorted node lines to the flowchart code
        self.flowchart_code += "\n" + "".join(node_lines)

    def generate_edges(self):
        # Regex to match edges, ensuring we capture nodes with various brackets or slashes
        edge_pattern = re.compile(
            r"([A-Za-z0-9_]+)\s*-->\s*(?:\|(.+?)\|)?\s*([A-Za-z0-9_]+)"
        )

        self.flowchart_code += "\n"
        for edge in self.edges:
            # Match the edge with the regex
            match = edge_pattern.match(edge)
            if match:
                from_node = match.group(1).strip()
                to_node = match.group(3).strip()  # The second node
                label = match.group(2) if match.group(2) else ""

                # Ensure we add the edge even if the nodes have extra symbols
                from_node_clean = re.sub(
                    r"(\[.*?\]|\(.*?\)|\{.*?\})", "", from_node
                ).strip()
                to_node_clean = re.sub(
                    r"(\[.*?\]|\(.*?\)|\{.*?\})", "", to_node
                ).strip()

                # Generate Python code for adding the edge
                if label:
                    label = label.replace('"', "")  # Removing extra quotes from label
                    # Fixing quotes for the label
                    self.flowchart_code += f'flowchart.add_edge("{from_node_clean}", "{to_node_clean}", "{label}")\n'
                else:
                    self.flowchart_code += (
                        f'flowchart.add_edge("{from_node_clean}", "{to_node_clean}")\n'
                    )

    def convert(self):
        self.parse_mermaid_code()
        self.generate_nodes()  # Ensure nodes are added before edges
        self.generate_edges()
        return self.flowchart_code


class Node:
    def __init__(self, id, description, shape):
        self.id = id
        self.description = description
        self.shape = shape
        self.edges = []

    def add_edge(self, target_node_id, condition=None):
        self.edges.append((target_node_id, condition))

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "shape": self.shape,
            "edges": self.edges,
        }

    @staticmethod
    def from_dict(node_dict):
        node = Node(node_dict["id"], node_dict["description"], node_dict["shape"])
        node.edges = node_dict["edges"]
        return node


class Flowchart:
    def __init__(self):
        self.nodes = {}

    def add_node(self, id, description, shape):
        if id not in self.nodes:
            self.nodes[id] = Node(id, description, shape)

    def add_edge(self, from_id, to_id, condition=None):
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].add_edge(to_id, condition)

    def to_dict(self):
        return {"nodes": {id: node.to_dict() for id, node in self.nodes.items()}}

    @staticmethod
    def from_dict(flowchart_dict):
        flowchart = Flowchart()
        for node_dict in flowchart_dict["nodes"].values():
            node = Node.from_dict(node_dict)
            flowchart.nodes[node.id] = node
        return flowchart

    def save_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r", encoding="utf-8") as f:
            flowchart_dict = json.load(f)
            return Flowchart.from_dict(flowchart_dict)

    def to_mermaid(self):
        mermaid_lines = ["flowchart TD"]
        for node in self.nodes.values():
            # Adjust node shape and description for Mermaid diagram
            if node.shape == "ellipse":
                mermaid_lines.append(
                    f'    {node.id}(["{node.description}"])'
                )  # Rounded rectangle
            elif node.shape == "box":
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Rectangle
            elif node.shape == "diamond":
                mermaid_lines.append(
                    f'    {node.id}{{"{node.description}"}}'
                )  # Diamond/decision shape
            elif node.shape == "parallelogram":
                mermaid_lines.append(
                    f'    {node.id}[/"{node.description}"/]'
                )  # Slanted parallelogram
            else:
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Default to box

            # Handle edges with or without conditions
            for target_id, condition in node.edges:
                if condition:
                    mermaid_lines.append(
                        f'    {node.id} -->|"{condition}"| {target_id}'
                    )
                else:
                    mermaid_lines.append(f"    {node.id} --> {target_id}")
        return "\n".join(mermaid_lines)

    # Link-based flow is the most common style used/generated by chatgpt
    # A link-based approach where each node is defined the first time it's encountered, with
    # 　subsequent references focusing solely on connecting links.
    def to_mermaid_link_based_flow(self):
        mermaid_lines = ["flowchart TD"]
        defined_nodes = set()  # Keep track of which nodes have been defined

        # Handle edges and inline node definitions when they appear for the first time
        for node in self.nodes.values():
            for target_id, condition in node.edges:
                line = ""

                # Define the source node inline if it's not already defined
                if node.id not in defined_nodes:
                    if node.shape == "ellipse":
                        line += f'{node.id}(["{node.description}"])'
                    elif node.shape == "box":
                        line += f'{node.id}["{node.description}"]'
                    elif node.shape == "diamond":
                        line += f'{node.id}{{"{node.description}"}}'
                    elif node.shape == "parallelogram":
                        line += f'{node.id}[/"{node.description}"/]'
                    else:
                        line += f'{node.id}["{node.description}"]'
                    defined_nodes.add(node.id)
                else:
                    line += f"{node.id}"

                # Define the target node inline if it's not already defined
                if target_id not in defined_nodes:
                    target_node = self.nodes[target_id]
                    if target_node.shape == "ellipse":
                        target_def = f'{target_id}(["{target_node.description}"])'
                    elif target_node.shape == "box":
                        target_def = f'{target_id}["{target_node.description}"]'
                    elif target_node.shape == "diamond":
                        target_def = f'{target_id}{{"{target_node.description}"}}'
                    elif target_node.shape == "parallelogram":
                        target_def = f'{target_id}[/"{target_node.description}"/]'
                    else:
                        target_def = f'{target_id}["{target_node.description}"]'
                    defined_nodes.add(target_id)
                else:
                    target_def = f"{target_id}"

                # Add condition if applicable
                if condition:
                    line += f' -->|"{condition}"| {target_def}'
                else:
                    line += f" --> {target_def}"

                # Append the line to the Mermaid lines
                mermaid_lines.append(f"    {line}")

        # Join the lines to form the complete Mermaid code
        return "\n".join(mermaid_lines)

    def to_mermaid_link_based_flow_shuffle(self):
        mermaid = self.to_mermaid_link_based_flow()
        return shuffle_mermaid(mermaid)

    def to_mermaid_link_based_flow_reverse(self):
        mermaid = self.to_mermaid_link_based_flow()
        return reverse_mermaid(mermaid)

    # Ｎode Edge Separation highlights the separation of node definition and linking
    # A structure that defines all nodes upfront, followed by establishing all connections, separating
    # the node definitions from their linking process.
    def to_mermaid_node_edge_separation(self):
        mermaid_lines = ["flowchart TD"]

        for node in self.nodes.values():
            # Adjust node shape and description for Mermaid diagram
            if node.shape == "ellipse":
                mermaid_lines.append(
                    f'    {node.id}(["{node.description}"])'
                )  # Rounded rectangle
            elif node.shape == "box":
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Rectangle
            elif node.shape == "diamond":
                mermaid_lines.append(
                    f'    {node.id}{{"{node.description}"}}'
                )  # Diamond/decision shape
            elif node.shape == "parallelogram":
                mermaid_lines.append(
                    f'    {node.id}[/"{node.description}"/]'
                )  # Slanted parallelogram
            else:
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Default to box

        # Handling edges with inline connection format
        for node in self.nodes.values():
            for target_id, condition in node.edges:
                if condition:
                    mermaid_lines.append(
                        f'    {node.id} -->|"{condition}"| {target_id}'
                    )
                else:
                    mermaid_lines.append(f"    {node.id} --> {target_id}")

        # Join edges and nodes into a more compact structure
        return "\n".join(mermaid_lines)

    def to_mermaid_node_edge_separation_shuffle(self):
        mermaid = self.to_mermaid_node_edge_separation()
        return shuffle_mermaid(mermaid)

    def to_mermaid_node_edge_separation_reverse(self):
        mermaid = self.to_mermaid_node_edge_separation()
        return reverse_mermaid(mermaid)

    # Sequential Connection Flow emphasizes the sequential handling of nodes and their connections
    # A sequential approach where each node and its connections are defined and linked immediately as they are encountered, processing nodes one by one in the order they appear.
    def to_mermaid_sequential_connection_flow(self):
        mermaid_lines = ["flowchart TD"]

        for node in self.nodes.values():
            # Adjust node shape and description for Mermaid diagram
            if node.shape == "ellipse":
                mermaid_lines.append(
                    f'    {node.id}(["{node.description}"])'
                )  # Rounded rectangle
            elif node.shape == "box":
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Rectangle
            elif node.shape == "diamond":
                mermaid_lines.append(
                    f'    {node.id}{{"{node.description}"}}'
                )  # Diamond/decision shape
            elif node.shape == "parallelogram":
                mermaid_lines.append(
                    f'    {node.id}[/"{node.description}"/]'
                )  # Slanted parallelogram
            else:
                mermaid_lines.append(
                    f'    {node.id}["{node.description}"]'
                )  # Default to box

            # Handle edges with or without conditions
            for target_id, condition in node.edges:
                if condition:
                    mermaid_lines.append(
                        f'    {node.id} -->|"{condition}"| {target_id}'
                    )
                else:
                    mermaid_lines.append(f"    {node.id} --> {target_id}")

        return "\n".join(mermaid_lines)

    def to_mermaid_sequential_connection_flow_shuffle(self):
        mermaid = self.to_mermaid_sequential_connection_flow()
        return shuffle_mermaid(mermaid)

    def to_mermaid_sequential_connection_flow_reverse(self):
        mermaid = self.to_mermaid_sequential_connection_flow()
        return reverse_mermaid(mermaid)

    def get_number_of_nodes(self):
        """Returns the number of nodes in the flowchart."""
        return len(self.nodes)

    def get_number_of_edges(self):
        """Returns the total number of edges in the flowchart."""
        return sum(len(node.edges) for node in self.nodes.values())

    def get_direct_successors(self, node_description):
        """Returns the direct successors (outgoing connections) of the given node by its description."""
        # Find the node by its description
        for node in self.nodes.values():
            if node.description == node_description:
                successors = [
                    self.nodes[target_id].description for target_id, _ in node.edges
                ]
                return successors
        return []

    def get_direct_predecessors(self, node_description):
        """Returns the direct predecessors (incoming connections) of the given node by its description."""
        predecessors = []
        for node in self.nodes.values():
            for target_id, _ in node.edges:
                if self.nodes[target_id].description == node_description:
                    predecessors.append(node.description)
        return predecessors

    def get_shortest_path_length(self, start_node_description, end_node_description):
        """Returns the number of edges in the shortest path between two nodes based on their descriptions."""

        # Find the node IDs for the given descriptions
        start_node_id = None
        end_node_id = None

        for node_id, node in self.nodes.items():
            if node.description == start_node_description:
                start_node_id = node_id
            if node.description == end_node_description:
                end_node_id = node_id

        if start_node_id is None or end_node_id is None:
            return -1  # One or both nodes with the given descriptions do not exist

        # Breadth-first search (BFS) to find the shortest path
        queue = deque([(start_node_id, 0)])  # (current_node_id, distance)
        visited = set()

        while queue:
            current_node_id, distance = queue.popleft()

            if current_node_id == end_node_id:
                return distance

            visited.add(current_node_id)

            for target_id, _ in self.nodes[current_node_id].edges:
                if target_id not in visited:
                    queue.append((target_id, distance + 1))

        return -1  # No path found

    def get_max_indegree(self):
        """Returns the maximum indegree (number of incoming edges) for any node."""
        indegree = {node_id: 0 for node_id in self.nodes}

        for node in self.nodes.values():
            for target_id, _ in node.edges:
                if target_id in indegree:
                    indegree[target_id] += 1

        return max(indegree.values())

    def get_max_outdegree(self):
        """Returns the maximum outdegree (number of outgoing edges) for any node."""
        return max(len(node.edges) for node in self.nodes.values())
