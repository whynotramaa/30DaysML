

### Entropy
    Measure of disorder / uncertainity
    E(s) = sum( -p * log(p) )
    for 2 class problem, min enropy = 0 and max entropy = 1

### Information Gain
    = E(parent) - {Weighted Entropy} * E(children)
    higher information gain = less entropy and column with highest entropy -> best one to split for decision trees

### Gini Impurity
    G = 1 - ( P~yes~^2 + P~no~^2)
    prefer gini over entropy if we have larger dataset as log is consuming to calculate

    but sometimes, entropy gives better split (rare but possible)

#### numerical data splitting ?

