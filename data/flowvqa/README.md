### FlowVQA Dataset Fix
We addressed flowchart rendering issues in the original FlowVQA dataset caused by improper use of backticks (``` ` ```) in `mermaid` fields. All backticks were replaced with single quotes (`'`) to ensure syntax correctness.

### Test Set
We randomly selected 200 flowcharts from the FlowVQA test set and tested the corresponding questions. Three flowcharts were selected twice due to the random seed.

### Updates by Zheng Xin
Further split the dataset described above into dev and test sets, ensuring no flowcharts overlap across the split.

#### Dev set 
116 questions over 13 flowcharts

| Task category | Number of questions |
|:--------------:|:------------------:|
| fact retrieval | 22 |
| applied scenarios | 22 |
| flow referential | 22 |
| topological | 50 |

| Flowchart type | Number of flowcharts |
|:--------------:|:--------------------:|
| wiki | 5 |
| instruct | 5 |
| code | 3 |

#### Test set
1862 questions over 184 flowcharts

| Task category | Number of questions |
|:--------------:|:------------------:|
| fact retrieval | 378 |
| applied scenarios | 376 |
| flow referential | 324 |
| topological | 784 |

| Flowchart type | Number of flowcharts |
|:--------------:|:--------------------:|
| wiki | 96 |
| instruct | 51 |
| code | 37 |
