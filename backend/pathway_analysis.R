#!/usr/bin/env Rscript

library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(pathview)
library(zip)

options(bitmapType = "cairo")
options(error = function() {
  traceback(2)
  print(geterrmessage())
  quit(save = "no", status = 1)
})

# ==========================
# Arguments
# ==========================
args <- commandArgs(trailingOnly = TRUE)
input_file  <- args[1]
out_root    <- args[2]
gene_format <- ifelse(length(args) >= 3, args[3], "SYMBOL")

home_dir <- getwd()
setwd(out_root)

organism <- "org.Hs.eg.db"

# ==========================
# Read input
# ==========================
df <- read.csv(input_file, header = TRUE)
colnames(df)[1:2] <- c("Gene", "Value")
df <- df %>% filter(!is.na(Gene) & !is.na(Value))

# ==========================
# Build z-score vector
# ==========================
gene_list <- df$Value
names(gene_list) <- df$Gene
gene_list <- sort(na.omit(gene_list), decreasing = TRUE)

# ==========================
# Mixed ID detection + mapping
# ==========================

ensembl_genes <- names(gene_list)[grepl("^ENSG", names(gene_list))]
symbol_genes  <- names(gene_list)[!grepl("^ENSG", names(gene_list))]

cat(sprintf("→ Found %d Ensembl and %d Symbol genes\n",
            length(ensembl_genes), length(symbol_genes)))

ids_all <- data.frame()

if (length(ensembl_genes) > 0) {
  ids_ens <- suppressWarnings(bitr(
    ensembl_genes, fromType="ENSEMBL", toType="ENTREZID", OrgDb=organism))
  ids_all <- bind_rows(ids_all, ids_ens)
}

if (length(symbol_genes) > 0) {
  ids_sym <- suppressWarnings(bitr(
    symbol_genes, fromType="SYMBOL", toType="ENTREZID", OrgDb=organism))
  ids_all <- bind_rows(ids_all, ids_sym)
}

ids_all <- ids_all %>%
  filter(!duplicated(ENTREZID)) %>%
  mutate(ENTREZID = as.character(ENTREZID))

map_rate <- nrow(ids_all) / length(unique(names(gene_list))) * 100
cat(sprintf("✅ Successfully mapped %.1f%% of genes to ENTREZ IDs\n", map_rate))

if (nrow(ids_all) == 0) stop("No gene IDs could be mapped!")

# ==========================
# Match back to df
# ==========================
df2 <- df[df$Gene %in% ids_all$ENSEMBL | df$Gene %in% ids_all$SYMBOL, ]
df2$ENTREZID <- ids_all$ENTREZID[
  match(df2$Gene, coalesce(ids_all$ENSEMBL, ids_all$SYMBOL))
]

kegg_gene_list <- df2$Value
names(kegg_gene_list) <- df2$ENTREZID
kegg_gene_list <- sort(na.omit(kegg_gene_list), decreasing = TRUE)

# ==========================
# Global symmetric color limits (99th percentile of |z|)
# ==========================
all_vals <- as.numeric(kegg_gene_list)
all_vals <- all_vals[is.finite(all_vals)]
zlim <- quantile(abs(all_vals), 0.99, na.rm = TRUE)
gene_limit <- c(-zlim, zlim)
cat("🎨 Global color limits (99th percentile):", round(gene_limit, 3), "\n")

# ==========================
# Run KEGG GSEA
# ==========================
kegg_organism <- "hsa"

kk2 <- gseKEGG(
  geneList      = kegg_gene_list,
  organism      = kegg_organism,
  nPerm         = 10000,
  minGSSize     = 3,
  maxGSSize     = 800,
  pvalueCutoff  = 0.05,
  pAdjustMethod = "none",
  keyType       = "ncbi-geneid"
)

kk2_df <- as.data.frame(kk2)
write.csv(kk2_df, file.path(out_root, "kegg_dataframe.csv"), row.names = FALSE)

if (nrow(kk2_df) == 0) {
  stop("⚠️ No significant KEGG pathways found.")
}

# ==========================
# Generate Pathway Images
# ==========================
for (i in seq_len(nrow(kk2_df))) {
  pid  <- kk2_df$ID[i]
  desc <- kk2_df$Description[i]
  cat("🧩 Processing pathway:", pid, "-", desc, "...\n")

  pathview(
    gene.data   = kegg_gene_list,
    pathway.id  = pid,
    species     = "hsa",
    gene.idtype = "ENTREZ",
    kegg.dir    = "/home/ec2-user/kegg_cache",
    kegg.native = TRUE,
    out.suffix  = pid,
    out.dir     = out_root,
    limit       = list(gene = gene_limit, cpd = c(0, 1)),
    rescale     = FALSE,
    low         = list(gene = "blue"),
    mid         = list(gene = "white"),
    high        = list(gene = "red"),
    node.sum    = "max",
    same.layer  = FALSE
  )
}

# ==========================
# Zip results
# ==========================
zip_file <- file.path(out_root, "pathway_results.zip")
png_files <- list.files(out_root, pattern="\\.png$", full.names=TRUE)
zip::zipr(zip_file, files=png_files)

unlink(list.files(out_root, pattern = "\\.png$", recursive = TRUE, full.names = TRUE))

setwd(home_dir)
cat("✅ Pathway analysis complete. Results saved to:", zip_file, "\n")
