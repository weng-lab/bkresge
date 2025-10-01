suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(RcppML)
    library(org.Hs.eg.db)
    library(clusterProfiler)
    # library(CoGAPS)
    library(sessioninfo)
})

# out_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80.rda"
# out_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_one_over.rda"
# out_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_specificity_02.rda"
out_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_percentile_01.rda"

### Notes
# The following code was adapted from the patternMarkers() function from `CoGAPS` to accept as input a matrix instead of a `CoGAPSObject`
# Additionally, matrix is not normalized before pattern marker calculation because RcppML already normalizes patterns
# Original function: https://rdrr.io/bioc/CoGAPS/man/patternMarkers-methods.html
patternMarkers <- function(featureLoadingsMatrix, sampleFactorsMatrix, threshold, axis, n) {
    ## check inputs to the function
    if (!(threshold %in% c("cut", "all", "flex"))) {
        stop("threshold must be either 'cut' or 'all' or 'flex")
    }
    if (!(axis %in% 1:2)) {
        stop("axis must be either 1 or 2")
    }
    # Validate the new argument 'n'
    if (!is.numeric(n) || n < 1) {
        stop("n must be a positive integer")
    }

    ## need to scale each row of the matrix of interest so that the maximum is 1
    resultMatrix <- if (axis == 1) featureLoadingsMatrix else stop("Invalid axis for this function.")
    library(scales)

    # normedMatrix <- t(apply(resultMatrix, 1, function(row) row / max(row)))
    normedMatrix <- resultMatrix

    ## default pattern marker calculation, each pattern has unit weight
    markerScores <- sapply(1:ncol(normedMatrix), function(patternIndex) {
        apply(normedMatrix, 1, function(row) {
            lp <- rep(0, ncol(normedMatrix))
            lp[patternIndex] <- 1
            return(sqrt(sum((row - lp)^2)))
        })
    })

    markerRanks <- apply(markerScores, 2, rank)
    colnames(markerScores) <- colnames(markerRanks) <- colnames(normedMatrix)

    ## Define the simplicityGENES function
    simplicityGENES <- function(As, Ps) {
        # rescale p's to have max 1
        pscale <- apply(Ps, 1, max)

        # rescale A in accordance with p's having max 1
        As <- sweep(As, 2, pscale, FUN = "*")

        # find the A with the highest magnitude
        Arowmax <- t(apply(As, 1, function(x) x / max(x)))

        # determine which genes are most associated with each pattern
        ssl <- matrix(NA, nrow = nrow(As), ncol = ncol(As), dimnames = dimnames(As))
        for (i in 1:ncol(As)) {
            lp <- rep(0, ncol(As))
            lp[i] <- 1
            ssl.stat <- apply(Arowmax, 1, function(x) sqrt(t(x - lp) %*% (x - lp)))
            ssl[order(ssl.stat), i] <- 1:length(ssl.stat)
        }

        return(ssl)
    }

    ## keep only a subset of markers for each pattern depending on the type of threshold
    if (threshold == "cut") {
        simGenes <- simplicityGENES(As = resultMatrix, Ps = sampleFactorsMatrix)
        patternMarkers <- list()
        nP <- ncol(simGenes)

        for (i in 1:nP) {
            sortSim <- names(sort(simGenes[, i], decreasing = FALSE))
            geneThresh <- min(which(simGenes[sortSim, i] > apply(simGenes[sortSim, ], 1, min)))
            markerGenes <- sortSim[1:geneThresh]
            markerGenes <- unique(markerGenes)
            patternMarkers[[i]] <- markerGenes
        }

        markersByPattern <- patternMarkers
    } else if (threshold == "all") # only the markers with the lowest scores
        {
            min_indices <- apply(markerScores, 1, which.min)
            patternsByMarker <- colnames(markerScores)[sapply(min_indices, `[`, 1)]
            markersByPattern <- sapply(colnames(markerScores),
                USE.NAMES = TRUE, simplify = FALSE,
                function(pattern) rownames(markerScores)[which(patternsByMarker == pattern)]
            )
        } else if (threshold == "flex") {
        # Assign genes to a pattern if they are among the three lowest values for that gene
        flexPatternsByGene <- apply(markerScores, 1, function(geneScores) {
            lowestThreePatterns <- order(geneScores)[1:2]
            return(colnames(markerScores)[lowestThreePatterns])
        })

        markersByPattern <- lapply(colnames(markerScores), function(pattern) {
            genesAssignedToPattern <- rownames(markerScores)[which(flexPatternsByGene == pattern, arr.ind = TRUE)]
            return(unique(genesAssignedToPattern))
        })
    }

    ## add TopRankedGenes
    topRankedGenes <- list()
    for (patternIndex in 1:ncol(markerScores)) {
        geneScores <- markerScores[, patternIndex]
        geneRanks <- markerRanks[, patternIndex]

        # Filter out genes with zero loading for this pattern
        nonZeroIndices <- which(featureLoadingsMatrix[, patternIndex] != 0)
        filteredGeneScores <- geneScores[nonZeroIndices]
        filteredGeneRanks <- geneRanks[nonZeroIndices]

        # Sort the filtered gene scores and ranks and take the top N
        sortedGeneIndices <- order(filteredGeneScores)
        topNIndices <- sortedGeneIndices[1:min(n, length(sortedGeneIndices))]

        # Extract the gene names for these top N indices
        topRankedGenes[[patternIndex]] <- rownames(markerScores)[nonZeroIndices[topNIndices]]
    }

    return(list(
        "PatternMarkers" = markersByPattern,
        "PatternMarkerRanks" = markerRanks,
        "PatternMarkerScores" = markerScores,
        "TopRankedGenes" = topRankedGenes
    ))
}

# Open log file (overwrite each run)
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_analysis_k_80.log"
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_analysis_k_80_one_over.log"
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_analysis_k_80_specificity_02.log"
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_analysis_k_80_percentile_01.log"
sink(log_file, append = FALSE, split = TRUE) # split=TRUE keeps console + file
options(width = 120)

log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

log_msg("===== Starting GO Analysis =====")

### load snRNA-seq data and assign to sce
snrna_seq_data_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
obj_name <- load(file = snrna_seq_data_path, verbose = TRUE) # returns the name of the loaded object
sce <- get(obj_name)
log_msg(sprintf(
    "Loaded snRNA-seq object '%s' with %d genes and %d cells",
    obj_name, nrow(sce), ncol(sce)
))

### load NMF results (object x)
path_for_x <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/nmf_x_k_80.rda"
load(file = path_for_x, verbose = TRUE)
log_msg(sprintf(
    "Loaded NMF object 'x' with %d genes (W) and %d patterns",
    nrow(x@w), ncol(x@w)
))

## mark technical patterns for discard
discard <- c() # update with known technical patterns
if (length(discard) > 0) {
    log_msg(sprintf("Discarding patterns: %s", paste(discard, collapse = ", ")))
} else {
    log_msg("No patterns marked for discard")
}

## set up marker gene detection
loads <- x@w
loads <- loads[, !colnames(loads) %in% discard]

no_contrib <- which(rowSums(loads) == 0)
log_msg(sprintf("Removing %d genes with no contribution across patterns", length(no_contrib)))
if (length(no_contrib) > 0) {
    log_msg(sprintf("Example genes removed: %s", paste(head(rownames(loads)[no_contrib]), collapse = ", ")))
}
loads <- loads[-no_contrib, ]

## filter mito genes
mito <- rownames(sce)[which(seqnames(sce) == "chrM")]
log_msg(sprintf("Filtering %d mitochondrial genes", length(mito)))
loads <- loads[!rownames(loads) %in% mito, ]

## keep only protein-coding genes
protein <- rownames(sce)[rowData(sce)$gene_type == "protein_coding"]
log_msg(sprintf("Filtering to %d protein-coding genes", length(protein)))
loads <- loads[rownames(loads) %in% protein, ]

# ## get marker genes per pattern
# log_msg("Running patternMarkers() to detect marker genes per pattern")
# marks <- patternMarkers(
#     loads,
#     x@h[rownames(x@h) %in% colnames(loads), ],
#     "all",
#     1,
#     100
# )

# genes <- marks$PatternMarkers
# names(genes) <- colnames(loads)
# log_msg(sprintf("Identified marker genes for %d patterns", length(genes)))

# ## get marker genes per pattern
# log_msg("Selecting genes based on > 1/num_genes contribution rule")

# num_genes <- nrow(loads)
# threshold <- 1 / num_genes
# log_msg(sprintf("Using threshold %.6f (1/%d genes)", threshold, num_genes))

# genes <- list()
# for (i in 1:ncol(loads)) {
#     pat_name <- colnames(loads)[i]
#     selected <- rownames(loads)[which(loads[, i] > threshold)]
#     genes[[pat_name]] <- selected
#     log_msg(sprintf(
#         "Pattern %s: selected %d genes above threshold",
#         pat_name, length(selected)
#     ))
# }

# log_msg(sprintf("Finished selecting genes for %d patterns", length(genes)))

# ## get marker genes per pattern (hybrid approach: specificity + top-N)
# # --- Config ---
# specificity_cutoff <- 0.2 # require at least 20% of a gene's load to fall in one pattern
# top_n <- 200 # then keep only the top 200 per pattern

# log_msg("Selecting genes based on specificity + top-N approach")
# log_msg(sprintf("Specificity cutoff: %.2f", specificity_cutoff))
# log_msg(sprintf("Top N per pattern: %d", top_n))

# # Normalize each gene<U+2019>s contributions across patterns to fractions
# row_fracs <- sweep(loads, 1, rowSums(loads), "/")

# genes <- list()
# for (i in 1:ncol(loads)) {
#     pat_name <- colnames(loads)[i]

#     # --- Step 1: Specificity filter ---
#     # Keep only genes where this pattern explains at least `specificity_cutoff` fraction
#     specific_genes <- which(row_fracs[, i] >= specificity_cutoff)

#     # --- Step 2: Within those, rank by absolute loading in this pattern ---
#     ranked <- order(loads[specific_genes, i], decreasing = TRUE)

#     # --- Step 3: Keep top-N ---
#     top_genes <- rownames(loads)[specific_genes][ranked][1:min(top_n, length(ranked))]

#     genes[[pat_name]] <- top_genes
#     log_msg(sprintf(
#         "Pattern %s: %d genes passed specificity cutoff, keeping top %d",
#         pat_name, length(specific_genes), length(top_genes)
#     ))
# }

# log_msg(sprintf("Finished selecting genes for %d patterns", length(genes)))

## get marker genes per pattern (percentile first, then specificity)
# --- Config ---
percentile_cutoff <- 0.90 # keep top 10% (per pattern, nonzero only)
specificity_cutoff <- 0.05 # require <U+2265>5% of a gene's total loading in one pattern

log_msg("Selecting genes with percentile + specificity approach")
log_msg(sprintf(
    "Percentile cutoff: %.2f (top %.0f%%)",
    percentile_cutoff, 100 * (1 - percentile_cutoff)
))
log_msg(sprintf("Specificity cutoff: %.2f", specificity_cutoff))

genes <- list()
row_fracs <- sweep(loads, 1, rowSums(loads), "/") # gene-wise fractions across patterns

for (i in 1:ncol(loads)) {
    pat_name <- colnames(loads)[i]
    weights <- loads[, i]

    # --- Step 1: remove zeros before percentile calc ---
    nonzero_weights <- weights[weights > 0]
    if (length(nonzero_weights) == 0) {
        genes[[pat_name]] <- character(0)
        log_msg(sprintf("Pattern %s: no nonzero genes", pat_name))
        next
    }

    # --- Step 2: percentile gate ---
    cutoff <- quantile(nonzero_weights, probs = percentile_cutoff, na.rm = TRUE)
    pct_genes <- names(weights)[weights >= cutoff]

    # --- Step 3: specificity gate ---
    spec_genes <- names(row_fracs[, i])[row_fracs[, i] >= specificity_cutoff]

    # --- Step 4: intersection ---
    sel_genes <- intersect(pct_genes, spec_genes)
    genes[[pat_name]] <- sel_genes

    log_msg(sprintf(
        "Pattern %s: %d genes passed percentile cutoff, %d passed specificity, %d kept",
        pat_name, length(pct_genes), length(spec_genes), length(sel_genes)
    ))
}

log_msg(sprintf("Finished selecting genes for %d patterns", length(genes)))



## run GO enrichment for each pattern
log_msg("Running GO enrichment analysis for each pattern")
go <- list()
for (i in 1:length(genes)) {
    pat_name <- names(genes)[i]
    log_msg(sprintf("Enriching GO for pattern %s with %d genes", pat_name, length(genes[[i]])))
    go[[i]] <- enrichGO(
        gene          = genes[[i]],
        universe      = rownames(loads),
        OrgDb         = org.Hs.eg.db,
        ont           = "ALL",
        pAdjustMethod = "BH",
        pvalueCutoff  = 0.05,
        qvalueCutoff  = 0.1,
        readable      = TRUE,
        keyType       = "SYMBOL"
    )
}

### save GO results
save(go, file = out_file)
log_msg(sprintf("Saved GO enrichment results to %s", out_file))

# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished GO Analysis =====")

sink()
