# =============================================================================
# DESeq2 Paired Analysis: miRNA copy counts, 8-week vs 16-week
# =============================================================================
# Install dependencies if needed:
#   install.packages("BiocManager")
#   BiocManager::install("DESeq2")
# =============================================================================

library(DESeq2)

# --- Load data ----------------------------------------------------------------
pre  <- read.csv("./data/Pg_8_weeks_infected_5_cols.csv",  check.names = FALSE)
post <- read.csv("./data/Pg_16_weeks_infected_5_cols.csv", check.names = FALSE)

# Drop the 'infected' label column
pre  <- pre[,  colnames(pre)  != "infected"]
post <- post[, colnames(post) != "infected"]

n_mice  <- nrow(pre)   # 10
n_mirna <- ncol(pre)   # 599

# --- Build count matrix -------------------------------------------------------
# DESeq2 expects: rows = features (miRNAs), cols = samples
# Column order: all pre samples then all post samples
count_mat <- t(rbind(pre, post))   # (599 miRNAs) x (20 samples)
storage.mode(count_mat) <- "integer"

# --- Sample metadata ----------------------------------------------------------
col_data <- data.frame(
  condition = factor(c(rep("pre", n_mice), rep("post", n_mice)),
                     levels = c("pre", "post")),
  mouse     = factor(rep(seq_len(n_mice), 2))
)
rownames(col_data) <- colnames(count_mat)

# --- DESeq2 paired design -----------------------------------------------------
dds <- DESeqDataSetFromMatrix(
  countData = count_mat,
  colData   = col_data,
  design    = ~ mouse + condition   # blocking on mouse = paired design
)

# Filter very low-count miRNAs (optional but recommended)
keep <- rowSums(counts(dds)) >= 10
dds  <- dds[keep, ]
cat(sprintf("miRNAs retained after low-count filter: %d\n", sum(keep)))

dds <- DESeq(dds)

# --- Extract results ----------------------------------------------------------
res <- results(dds,
               contrast  = c("condition", "post", "pre"),
               alpha     = 0.05)       # used for independent filtering threshold

res_df <- as.data.frame(res)
res_df <- res_df[order(res_df$padj, na.last = TRUE), ]

cat(sprintf("Significant miRNAs (padj < 0.05): %d\n",
            sum(res_df$padj < 0.05, na.rm = TRUE)))

write.csv(res_df, "deseq2_group_results.csv", quote = FALSE)
cat("Saved: deseq2_group_results.csv\n")

# --- Normalized counts for downstream Python script --------------------------
norm_counts <- counts(dds, normalized = TRUE)
write.csv(norm_counts, "deseq2_normalized_counts.csv", quote = FALSE)
cat("Saved: deseq2_normalized_counts.csv\n")
cat("  (Feed this into loo_crawford.py for per-mouse analysis)\n")
