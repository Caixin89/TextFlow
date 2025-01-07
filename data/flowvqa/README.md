### FlowVQA Dataset Fix
We addressed flowchart rendering issues in the original FlowVQA dataset caused by improper use of backticks (``` ` ```) in `mermaid` fields. All backticks were replaced with single quotes (`'`) to ensure syntax correctness.

### Test Set
We randomly selected 200 flowcharts from the FlowVQA test set and tested the corresponding questions. Three flowcharts were selected twice due to the random seed.