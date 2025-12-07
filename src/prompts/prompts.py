def load_vqa_prompt(question):
    return f"Question: {question}\nAnswer:"


def load_textualizer_prompt(output_type):
    if output_type == "mermaid":
        return """Generate the Mermaid code for the provided flowchart.

Here is an example:
```mermaid
flowchart TD
    A(["Start"]) --> B[/"Receive 'arr' and 'n'"/]
    B --> C["Initialize loop index 'i' to 0"]
    C --> D{"Check if arr[i] == i"}
    D -->|"Yes"| E[/"Return index 'i' as fixed point"/]
    E --> F(["End"])
    D -->|"No"| G["Increment 'i'"]
    G --> H{"i < n"}
    H -->|"Yes"| D
    H -->|"No"| I[/"Return -1 as no fixed point found"/]
    I --> F
```"""
    elif output_type == "graphviz":
        return """Generate the Graphviz code for the provided flowchart.

Here is an example:
```dot
digraph G {
    A [label="Start" shape=ellipse];
    B [label="Receive 'arr' and 'n'" shape=parallelogram];
    C [label="Initialize loop index 'i' to 0" shape=box];
    D [label="Check if arr[i] == i" shape=diamond];
    E [label="Return index 'i' as fixed point" shape=parallelogram];
    F [label="End" shape=ellipse];
    G [label="Increment 'i'" shape=box];
    H [label="i < n" shape=diamond];
    I [label="Return -1 as no fixed point found" shape=parallelogram];

    A -> B;
    B -> C;
    C -> D;
    D -> E [label="Yes"];
    E -> F;
    D -> G [label="No"];
    G -> H;
    H -> D [label="Yes"];
    H -> I [label="No"];
    I -> F;
}
```"""
    elif output_type == "plantuml":
        return """Generate the PlantUML code for the provided flowchart.

Here is an example:
```plantuml
@startuml
start
:Receive 'arr' and 'n';
:Initialize loop index 'i' to 0;

while (i < n?) is (Yes)
    if (Check if arr[i] == i?) then (Yes)
        :Return index 'i' as fixed point;
        stop
    else (No)
        :Increment 'i';
    endif
endwhile (No)
:Return -1 as no fixed point found;
stop
@enduml
```"""
    else:
        raise ValueError(f"Unsupported output type: {output_type}")


def load_reasoner_prompt(question, represenation):
    return f"{represenation}\n\nQuestion: {question}\nAnswer:"


def load_evaluation_prompt(question, response, label):
    prompt = f'''You are acting as a strict evaluation judge for a Flowchart VQA task.

You are given:
1. A QUESTION about the behavior or outcome implied by a flowchart.
2. A GROUND TRUTH ANSWER written by a human.
3. A MODEL ANSWER produced by a system under evaluation.

Your task is to determine whether the MODEL ANSWER is SEMANTICALLY EQUIVALENT to the GROUND TRUTH ANSWER with respect to the QUESTION.

Evaluation Rules:
- Judge only based on the QUESTION and the two ANSWERS. You do NOT reconstruct or imagine the flowchart.
- Paraphrasing is allowed. If the MODEL ANSWER expresses the same meaning, count it as Correct.
- If there is any meaningful difference in condition, branching, requirement, or outcome, mark it as Incorrect.
- Ignore minor stylistic or wording differences that do not change meaning.
- If the model answer omits an essential condition present in the ground truth, it is Incorrect.
- If the model answer adds an unsupported condition or extra incorrect logic, it is Incorrect.
- When uncertain, choose the stricter option (prefer “Incorrect”).
- Follow the output format exactly.

You must respond with a single JSON object and nothing else:

{{
  "verdict": "Correct" | "Incorrect",
  "explanation": "1–3 sentences explaining your decision."
}}

Now judge the following:

QUESTION:
{question}

GROUND TRUTH ANSWER:
{label}

MODEL ANSWER:
{response}'''

    return prompt
