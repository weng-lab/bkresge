# library("SpatialExperiment")
# library("BayesSpace")
# # Load the BayesSpace clustering output for inspection
# load("/zata/zippy/kresgeb/psych_screen/output/bayes_space/spe_clustered.Rdata", verbose = TRUE)


library(mclust)

# Ground truth: 3 balanced clusters
true_labels <- c(rep("A", 5), rep("B", 5), rep("C", 5))

# 1. Perfect match (baseline)
pred_1 <- c(rep("A", 5), rep("B", 5), rep("C", 5))

# 2. Refined clustering: "B" split into B1 and B2
pred_2 <- c(rep("A", 5), rep("B1", 3), rep("B2", 2), rep("C", 5))

# 3. Merged clustering: "A" and "B" merged into one group AB
pred_3 <- c(rep("AB", 10), rep("C", 5))

# 4. Slightly noisy clustering (shuffle a few points)
pred_4 <- c(
    "A", "A", "A", "B", "A", # slight noise in A
    "B", "B", "C", "B", "B", # one C in B
    "C", "C", "C", "C", "C"
)

# 5. Shuffled names and A to B, B to C, C to A
pred_5 <- c(rep("B", 5), rep("C", 5), rep("A", 5))


# Compute and print ARIs
cat("ARI for perfect match:", adjustedRandIndex(true_labels, pred_1), "\n")
cat("ARI for refined clustering (split B):", adjustedRandIndex(true_labels, pred_2), "\n")
cat("ARI for merged clustering (A+B):", adjustedRandIndex(true_labels, pred_3), "\n")
cat("ARI for noisy clustering:", adjustedRandIndex(true_labels, pred_4), "\n")
cat("ARI for shuffled names (A to B, B to C, C to A):", adjustedRandIndex(true_labels, pred_5), "\n")

# Print alignment tables
cat("\nContingency tables:\n\n")
print(table(True = true_labels, Predicted = pred_1))
print(table(True = true_labels, Predicted = pred_2))
print(table(True = true_labels, Predicted = pred_3))
print(table(True = true_labels, Predicted = pred_4))
print(table(True = true_labels, Predicted = pred_5))
