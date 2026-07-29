# Stage 8A protocol clarification 002

Date: 2026-07-23  
Classification: non-performance clarification  
Effect on frozen analyses: none

## Normalized-linear and cosine labels

The sealed learned-representation grid contains separate `linear` and `cosine`
labels for the 256-dimensional MLP hidden representation. In the released
implementation, `scripts/stage8a_common.py::kernel_matrix` routes both labels to
`normalized_linear`. That function divides the inner-product Gram matrix by the
outer product of the sample-vector norms. For non-zero representations, the two
labels therefore produce the same Gram matrix.

Accordingly, the sealed search contains:

- 19 registered candidate labels representing 18 unique Gram constructions;
- 76 registered candidate-and-\(C\) labels representing 72 unique
  Gram-and-\(C\) configurations.

The duplicate label is retained because candidate order is part of the sealed
protocol and selection audit. Selection updates only for a strictly larger
validation score, so an exact tie retains the earlier label. No candidate,
result row, selection outcome, performance value, interval, figure, or
scientific conclusion was removed or recomputed for this clarification.

This document records an implementation alias identified during manuscript
audit. It is not a retrospective amendment to the experimental protocol.
