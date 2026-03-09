#!/usr/bin/env Rscript
library(edgeR)
library(limma)

# Get arguments
args <- commandArgs(trailingOnly = TRUE)
count_data_file    <- args[1]  # counts file
sample_metadata_file <- args[2]  # metadata file
out_file           <- args[3]  # output file
design_factor      <- args[4]  # column in metadata (e.g. "Treatment")
meta_index_col     <- args[5]  # column in metadata that is sample ID (e.g. "SampleName")

# Detect separator based on file extension
get_sep <- function(fname) {
  if (grepl("\\.csv$", fname, ignore.case = TRUE)) {
    return(",")
  } else {
    return("\t")
  }
}

count_sep <- get_sep(count_data_file)
meta_sep  <- get_sep(sample_metadata_file)

# LOAD DATA ###################################################
count_data <- read.delim(count_data_file, sep = count_sep, header = TRUE, stringsAsFactors = FALSE)

# Assume first column is gene IDs
gene_ids <- count_data[,1]
gene_ids <- make.unique(as.character(gene_ids))
rownames(count_data) <- gene_ids
count_data <- count_data[,-1]

# Load metadata
sample_data <- read.delim(sample_metadata_file, sep = meta_sep, header = TRUE, stringsAsFactors = FALSE)

# Set the row names of metadata to the chosen index column
if (!(meta_index_col %in% colnames(sample_data))) {
  stop(paste("FATAL ERROR: Metadata does not contain index column", meta_index_col))
}
rownames(sample_data) <- sample_data[[meta_index_col]]
sample_data[[meta_index_col]] <- NULL  # drop index col now redundant

# Sort both to align
sample_data <- sample_data[order(rownames(sample_data)), , drop=FALSE]
count_data  <- count_data[,order(colnames(count_data))]

# Ensure sample IDs match
if (!isTRUE(all.equal(rownames(sample_data), colnames(count_data)))) {
  msg <- all.equal(rownames(sample_data), colnames(count_data))
  stop(paste("FATAL ERROR: Sample IDs mismatch:", msg))
}

# Ensure design factor exists
if (!(design_factor %in% colnames(sample_data))) {
  stop(paste("FATAL ERROR: Metadata must contain column", design_factor))
}

# Build EdgeR object
y <- DGEList(counts = count_data, samples = sample_data, group = sample_data[[design_factor]])

# FILTER
keep.exprs <- filterByExpr(y, group = y$samples[[design_factor]])
x <- y[keep.exprs,, keep.lib.sizes = FALSE]

# FILTERED COUNTS
cts <- as.data.frame(x$counts)
cts$Geneid <- rownames(cts)
cts <- cts[, c("Geneid", setdiff(colnames(cts), "Geneid"))]

# Write output
write.table(cts, file = out_file, sep = "\t", quote = FALSE, row.names = FALSE)



