#!/usr/bin/env Rscript

library(KEGGREST)

cache_dir <- "/home/ec2-user/kegg_cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# List all human pathways
kegg_list <- KEGGREST::keggList("pathway", "hsa")
path_ids <- sub("path:hsa", "", names(kegg_list))
print(path_ids)
message("Found ", length(path_ids), " human KEGG pathways.")

# Helper: download XML + PNG manually
download_if_missing <- function(pid, dest) {
  xml_file <- file.path(dest, paste0(pid, ".xml"))
  png_file <- file.path(dest, paste0(pid, ".png"))
  
  if (file.exists(xml_file) && file.exists(png_file)) {
    message("Skipping hsa", pid, " (already cached)")
    return(invisible(TRUE))
  }
  
  message("Downloading hsa", pid, " ...")
  xml_url <- paste0("https://rest.kegg.jp/get/", pid, "/kgml")
  png_url <- paste0("https://rest.kegg.jp/get/", pid, "/image")
  
  tryCatch({
    if (!file.exists(xml_file))
      download.file(xml_url, xml_file, quiet = TRUE)
    if (!file.exists(png_file))
      download.file(png_url, png_file, quiet = TRUE)
  }, error = function(e) {
    message("⚠️ Failed: hsa", pid, " — ", e$message)
  })
}

# Loop through all pathways
for (pid in path_ids) {
  download_if_missing(pid, cache_dir)
  Sys.sleep(0.1)  # be polite to KEGG
}

message("✅ Finished downloading all KEGG human pathways to ", cache_dir)
