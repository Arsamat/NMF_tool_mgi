#!/usr/bin/env Rscript
library(edgeR)
library(limma)
library(biomaRt)

# Gene annotation: Ensembl ID -> SYMBOL, etc. (from RNA_Analysis_Complete_Final.Rmd)
build_gene_table_from_ensembl <- function(ensembl_ids, mySpecies = "human") {
  ds <- if(mySpecies == "mouse") "mmusculus_gene_ensembl" else "hsapiens_gene_ensembl"
  mart <- tryCatch(
    biomaRt::useEnsembl(biomart = "ensembl", dataset = ds, mirror = "useast"),
    error = function(e) biomaRt::useEnsembl(biomart = "ensembl", dataset = ds, mirror = "asia")
  )
  annot_lookup <- biomaRt::getBM(
    mart = mart,
    attributes = c("wikigene_description", "ensembl_gene_id", "entrezgene_id",
                   "gene_biotype", "chromosome_name", "external_gene_name")
  )
  annot_lookup <- annot_lookup[which(annot_lookup$ensembl_gene_id %in% ensembl_ids), ]
  annot_lookup <- annot_lookup[match(ensembl_ids, annot_lookup$ensembl_gene_id), ]
  gene_data <- data.frame(
    GENENAME = annot_lookup$wikigene_description,
    ENTREZID = annot_lookup$entrezgene_id,
    BIOTYPE = annot_lookup$gene_biotype,
    SYMBOL = annot_lookup$external_gene_name,
    CHROMOSOME = annot_lookup$chromosome_name,
    ENSEMBLID = ensembl_ids,
    row.names = ensembl_ids
  )
  gene_data[gene_data == ""] <- "NA"
  gene_data[is.na(gene_data)] <- "NA"
  gene_data
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript deg_analysis.R <counts_file> <metadata_file> <output_file> [species] [group_a_file] [group_b_file]")
}
cat("Getting arguments...\n")
count_file     <- args[1]
metadata_file  <- args[2]
output_file    <- args[3]
species        <- if (length(args) >= 4) args[4] else "human"

cat("[DEG] Starting analysis:", count_file, "\n")
cat("[DEG] Species:", species, "\n")

get_sep <- function(fname) {
  if (grepl("\\.csv$", fname, ignore.case = TRUE)) return(",")
  return("\t")
}

# Load counts: genes as rows, samples as columns
count_data <- read.delim(count_file, sep = get_sep(count_file), header = TRUE,
                         stringsAsFactors = FALSE, check.names = FALSE)
gene_ids <- count_data[, 1]
rownames(count_data) <- gene_ids
count_data <- count_data[, -1]

# Ensure numeric matrix, no NA/NaN/Inf (can cause calcNormFactors to fail)
count_data <- as.matrix(count_data)
storage.mode(count_data) <- "double"
count_data[!is.finite(count_data) | count_data < 0] <- 0

# Load metadata: at least SampleName; Group optional if group_a/group_b files provided
sample_data <- read.delim(metadata_file, sep = get_sep(metadata_file), header = TRUE,
                          stringsAsFactors = FALSE, check.names = FALSE)

if (!"SampleName" %in% colnames(sample_data)) {
  stop("Metadata must contain 'SampleName' column")
}

rownames(sample_data) <- sample_data$SampleName

# Align samples (GroupA and GroupB can each contain samples from different original metadata groups)
common_samples <- intersect(rownames(sample_data), colnames(count_data))
if (length(common_samples) == 0) {
  stop("No matching sample IDs between metadata and count matrix")
}

sample_data <- sample_data[common_samples, , drop = FALSE]
count_data  <- count_data[, common_samples, drop = FALSE]

# Ensure Group is a factor with GroupB as reference (so logFC = GroupA - GroupB)
sample_data$Group <- factor(sample_data$Group, levels = c("GroupB", "GroupA"))
cat("[DEG] Group levels:", levels(sample_data$Group), "\n")


# Require replication: at least 2 samples per group for DE analysis
n_per_group <- table(sample_data$Group)
if (any(n_per_group < 2)) {
  stop("DEG analysis requires at least 2 samples per group. Current: ",
       paste(names(n_per_group), "=", n_per_group, collapse = ", "))
}

# DGEList
y <- DGEList(counts = count_data, samples = sample_data, group = sample_data$Group)

# Filter lowly expressed genes
keep <- filterByExpr(y, group = y$samples$Group)
x <- y[keep, , keep.lib.sizes = FALSE]
cat("[DEG] Genes after filterByExpr:", nrow(x), "/", nrow(y), "\n")

if (nrow(x) == 0) {
  stop("No genes passed filterByExpr. Check your count matrix.")
}

# TMM normalization (can fail with sparse/small data; fallback to RLE or none)
x <- tryCatch(
  calcNormFactors(x, method = "TMM"),
  error = function(e) {
    cat("[DEG] TMM failed (", conditionMessage(e), "), trying RLE...\n")
    tryCatch(
      calcNormFactors(x, method = "RLE"),
      error = function(e2) {
        cat("[DEG] RLE failed, using no normalization (norm.factors=1)\n")
        x$samples$norm.factors <- rep(1, ncol(x))
        x
      }
    )
  }
)

# Design and voom
design <- model.matrix(~ Group + Run, data = x$samples)
v <- voom(x, design, plot = FALSE)

# Fit and eBayes
fit <- lmFit(v, design)
fit <- eBayes(fit)

# Full results table
top_table <- topTable(fit, coef = 2, number = Inf, sort.by = "P", adjust.method = "BH")

# Add gene ID as column
top_table$gene <- rownames(top_table)

# Ensembl -> symbol annotation (optional; may fail if Ensembl unreachable)
cat("[DEG] Fetching gene annotations from Ensembl...\n")
gene_annot <- tryCatch({
  build_gene_table_from_ensembl(rownames(top_table), species)
}, error = function(e) {
  message("Gene annotation skipped: ", conditionMessage(e))
  NULL
})
if (!is.null(gene_annot)) {
  cat("[DEG] Gene annotation complete.\n")
}

if (!is.null(gene_annot)) {
  top_table$SYMBOL   <- gene_annot[rownames(top_table), "SYMBOL"]
  top_table$GENENAME <- gene_annot[rownames(top_table), "GENENAME"]
  top_table <- top_table[, c("gene", "SYMBOL", "GENENAME", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
} else {
  top_table$SYMBOL   <- top_table$gene
  top_table$GENENAME <- NA_character_
  top_table <- top_table[, c("gene", "SYMBOL", "GENENAME", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
}

write.csv(top_table, output_file, row.names = FALSE)
cat("[DEG] Done. Wrote", nrow(top_table), "genes to", output_file, "\n")

# --- Top-30 heatmap: log2 CPM, genes = top 30 by expression variability ---
out_dir <- dirname(output_file)
heatmap_matrix_path <- file.path(out_dir, "heatmap_matrix.csv")
heatmap_annotation_path <- file.path(out_dir, "heatmap_annotation.csv")
cpm_mat <- edgeR::cpm(x, log = TRUE)
gene_var <- apply(cpm_mat, 1, var)
genes_top30 <- names(sort(gene_var, decreasing = TRUE))[seq_len(min(30, nrow(cpm_mat)))]
genes_top30 <- genes_top30[!is.na(genes_top30) & genes_top30 != ""]
if (length(genes_top30) > 0) {
  hm <- cpm_mat[genes_top30, , drop = FALSE]
  rownames(hm) <- if ("SYMBOL" %in% colnames(top_table)) {
    syms <- top_table[genes_top30, "SYMBOL"]
    ifelse(!is.na(syms) & syms != "NA", as.character(syms), genes_top30)
  } else genes_top30
  # Hierarchical clustering on genes (rows): order so similar genes are adjacent
  if (nrow(hm) >= 2) {
    gene_dist <- dist(hm, method = "euclidean")
    gene_hc <- hclust(gene_dist, method = "complete")
    hm <- hm[gene_hc$order, , drop = FALSE]
  }
  write.csv(hm, heatmap_matrix_path)
  anno <- data.frame(SampleName = colnames(x), Group = as.character(x$samples$Group))
  write.csv(anno, heatmap_annotation_path, row.names = FALSE)
  cat("[DEG] Wrote top-30 (by variability) log2 CPM heatmap matrix (genes clustered) and sample annotation.\n")
}

# --- GSEA (Hallmark) ---
gsea_path <- file.path(out_dir, "gsea_results.csv")
gsea_ok <- FALSE
if (requireNamespace("clusterProfiler", quietly = TRUE) &&
    requireNamespace("msigdbr", quietly = TRUE) &&
    (requireNamespace("org.Hs.eg.db", quietly = TRUE) || requireNamespace("org.Mm.eg.db", quietly = TRUE))) {
  tryCatch({
    library(clusterProfiler)
    library(msigdbr)
    orgdb <- if (species == "mouse") org.Mm.eg.db::org.Mm.eg.db else org.Hs.eg.db::org.Hs.eg.db
    # Ranked list: logFC by Ensembl ID, sorted decreasing
    ranked <- setNames(top_table$logFC, top_table$gene)
    ranked <- ranked[order(ranked, decreasing = TRUE)]
    # Strip version suffix for Ensembl->Entrez mapping (e.g. ENSG00000000003.14 -> ENSG00000000003)
    ensembl_base <- sub("\\.[0-9]+$", "", names(ranked))
    map_df <- AnnotationDbi::select(orgdb, keys = unique(ensembl_base), keytype = "ENSEMBL", columns = "ENTREZID")
    map_df <- map_df[!is.na(map_df$ENTREZID) & !duplicated(map_df$ENSEMBL), ]
    # Map ranked to Entrez (use first match per Ensembl if multiple Entrez)
    idx <- match(ensembl_base, map_df$ENSEMBL)
    r2 <- ranked[!is.na(idx)]
    names(r2) <- map_df$ENTREZID[idx[!is.na(idx)]]
    # GSEA requires unique gene names: collapse duplicate Entrez by keeping max |logFC| per gene
    r2 <- r2[!is.na(names(r2)) & names(r2) != ""]
    if (length(r2) > 0) {
      r2_df <- data.frame(ENTREZID = names(r2), logFC = as.numeric(r2), stringsAsFactors = FALSE)
      r2_agg <- aggregate(logFC ~ ENTREZID, data = r2_df, function(x) x[which.max(abs(x))])
      r2 <- setNames(r2_agg$logFC, r2_agg$ENTREZID)
      r2 <- r2[order(r2, decreasing = TRUE)]
    }
    if (length(r2) >= 10) {
      species_name <- if (species == "mouse") "Mus musculus" else "Homo sapiens"
      h_t2g <- msigdbr::msigdbr(species = species_name, category = "H")[, c("gs_name", "entrez_gene")]
      set.seed(1)
      gsea_result <- clusterProfiler::GSEA(geneList = r2, TERM2GENE = h_t2g,
                                           minGSSize = 10, maxGSSize = 500, pvalueCutoff = 0.25, verbose = FALSE)
      if (!is.null(gsea_result) && nrow(as.data.frame(gsea_result)) > 0) {
        write.csv(as.data.frame(gsea_result), gsea_path, row.names = FALSE)
        gsea_ok <- TRUE
        cat("[DEG] Wrote GSEA Hallmark results.\n")
      }
    }
  }, error = function(e) message("[DEG] GSEA skipped: ", conditionMessage(e)))
}
if (!gsea_ok) cat("[DEG] GSEA not run (packages missing or no results).\n")
