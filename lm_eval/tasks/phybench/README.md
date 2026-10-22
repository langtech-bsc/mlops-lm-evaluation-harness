# PHYBench

This task evaluates the subset of `Eureka-Lab/PHYBench` that contains reference
answers. Rows without an answer are filtered out.

The official benchmark reports accuracy and Expression Edit Distance (EED).
The EED implementation is not a dependency of lm-evaluation-harness, so this
task reports normalized exact match and `math_verify` symbolic equivalence.
These metrics should not be compared directly with official EED scores.
