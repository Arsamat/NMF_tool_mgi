library(edgeR)   
library(limma)

args <- commandArgs(trailingOnly = TRUE)
count_data_file    <- args[1]  # counts file
sample_metadata_file <- args[2]  # metadata file
out_file           <- args[3]  # output file
design_factor      <- args[4]  # column in metadata (e.g. "Treatment")
meta_index_col     <- args[5]  # column in metadata that is sample ID (e.g. "SampleName")
hvg <- as.integer(args[6])
batch              <- as.logical(args[7])  # TRUE/FALSE
batch_column       <- args[8]              # column in metadata used as batch
batch_include      <- args[9:length(args)] # extra covariates for design

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

sample_data[[design_factor]] <- factor(sample_data[[design_factor]])

# Build EdgeR object
y <- DGEList(counts = count_data, samples = sample_data, group = sample_data[[design_factor]])

# FILTER
keep.exprs <- filterByExpr(y, group = y$samples[[design_factor]])
x <- y[keep.exprs,, keep.lib.sizes = FALSE]

# Normalize
x <- calcNormFactors(x, method = "TMM")

# TMM-normalized logCPM
logCPM <- edgeR::cpm(x, log = TRUE, prior.count = 0.1, normalized.lib.sizes = TRUE)

# ---------------------------
# CONDITIONAL BATCH CORRECTION
# ---------------------------
if (batch) {
  if (!(batch_column %in% colnames(x$samples))) {
    stop(paste("FATAL ERROR: Metadata does not contain batch column", batch_column))
  }
  
  # Build design matrix
  if (length(batch_include) > 0) {
    formula_str <- paste("~", paste(batch_include, collapse = " + "))
    mod <- model.matrix(as.formula(formula_str), data = x$samples)
  } else {
    mod <- model.matrix(~1, data = x$samples)
  }
  
  lcpm_rbe <- limma::removeBatchEffect(
    logCPM,
    batch  = x$samples[[batch_column]],
    design = mod
  )
} else {
  lcpm_rbe <- logCPM
}

# Pick top H most variable genes 
gene_vars <- apply(lcpm_rbe, 1, var)
ranked_idx <- order(gene_vars, decreasing = TRUE)

H <- 5000
top_genes <- rownames(lcpm_rbe)[ranked_idx[1:H]]

lcpm_rbe_top <- lcpm_rbe[top_genes, ]

# Force positive for NMF
X_nmf <- lcpm_rbe_top - min(lcpm_rbe_top) + 1e-6

# Ensure sample order
X_nmf <- X_nmf[, rownames(x$samples)]

cat("Final matrix dimensions:\n")
print(dim(X_nmf))
print(rownames(X_nmf)[1:5])
print(colnames(X_nmf)[1:5])

# Save as table
X_nmf_df <- as.data.frame(X_nmf)
X_nmf_df$Geneid <- rownames(X_nmf)
X_nmf_df <- X_nmf_df[, c("Geneid", setdiff(colnames(X_nmf_df), "Geneid"))]

write.table(X_nmf_df,
            file = out_file,
            sep = "\t",
            quote = FALSE,
            row.names = FALSE)

