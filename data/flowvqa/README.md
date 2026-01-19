### FlowVQA Dataset Fix
We addressed flowchart rendering issues in the original FlowVQA dataset caused by improper use of backticks (``` ` ```) in `mermaid` fields. All backticks were replaced with single quotes (`'`) to ensure syntax correctness.

### Test Set
We randomly selected 200 flowcharts from the FlowVQA test set and tested the corresponding questions. Three flowcharts were selected twice due to the random seed.

### Updates by Zheng Xin
Further split the dataset descrived above into dev and test sets, ensuring 2 flowcharts overlap across the split:
    Dev set - 116 questions over 16 flowcharts 
        - 22 T1 (fact retrieval)
        - 22 T2 (applied scenarios)
        - 22 T3 (flow referential)
        - 50 T4 (topological)
        - 5 wiki flowcharts
        - 5 instruct flowcharts
        - 3 code flowcharts
    Test set - 1862 questions over 184 flowcharts
        - 378 T1 (fact retrieval)
        - 376 T2 (applied scenarios)
        - 324 T3 (flow referential)
        - 784 T4 (topological)
        - 96 wiki flowcharts
        - 51 instruct flowcharts
        - 37 code flowcharts
