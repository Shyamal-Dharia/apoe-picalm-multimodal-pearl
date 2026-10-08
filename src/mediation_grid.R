## ======================================================================
## Omnibus mediation grid: gene -> sMRI mediator (med_*) -> outcome (y_*)
## mediation::mediate, quasi-Bayesian Monte Carlo (10,000 sims), gene coded
## 0 = N, 1 = A+P-, 2 = A+P+ and contrasted as 2 vs 0. ACME p-values are
## FDR-corrected across all mediator-outcome pairs.
##
## Run from the repository root:
##   MEDIATION_CORES=60 Rscript src/mediation_grid.R            # all outcomes
##   MEDIATION_CORES=60 Rscript src/mediation_grid.R 1 16       # outcomes 1-16 only
## One CSV per outcome is written to results/mediation/grid_parts/ and
## outcomes that already have a part file are skipped, so runs can resume or
## be split across jobs. When every outcome is done, the parts are combined
## and FDR-corrected. Each model has its own seed, so results do not depend
## on the number of cores.
## ======================================================================

suppressPackageStartupMessages({
  library(mediation)
  library(tidyverse)
  library(parallel)
})

data_path     <- "data/mediation_input_n64.csv"   # from src/build_mediation_input.py
sample_path   <- "data/participants_n64.csv"   # from src/select_participants.py
parts_dir     <- "results/mediation/grid_parts"
out_all       <- "results/mediation/mediation_grid_all_pairs.csv"
out_sig       <- "results/mediation/mediation_grid_significant_acme.csv"
n_sims        <- 10000
base_seed     <- 2026L
n_cores       <- as.integer(Sys.getenv("MEDIATION_CORES", Sys.getenv("SLURM_CPUS_PER_TASK", "1")))

dir.create(parts_dir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## 1. Load data (analysis sample only)
## -----------------------------
dat <- read.csv(data_path, check.names = TRUE)
sample_ids <- read.csv(sample_path)$participant_id
dat <- dat[dat$participant_id %in% sample_ids, ]
stopifnot(nrow(dat) == length(sample_ids))

if (is.factor(dat$gene)) dat$gene <- as.numeric(as.character(dat$gene))
message("n = ", nrow(dat), " (gene: ", paste(names(table(dat$gene)), table(dat$gene), sep = "=", collapse = ", "), ")")

## -----------------------------
## 2. Variables
## -----------------------------
treat_var <- "gene"
med_vars  <- grep("^med_", names(dat), value = TRUE)
med_vars  <- med_vars[!grepl("^med_(EEG|fMRI)_", med_vars)]   # sMRI mediators only
y_vars    <- grep("^y_", names(dat), value = TRUE)

covars <- c("age", "sex", "education", "learning_deficits", "HSV_r", "BMI",
            "hypertension", "thyroid_diseases", "other_diseases", "dementia_history_parents")
stopifnot(all(covars %in% names(dat)))
dat[covars] <- lapply(dat[covars], function(x) if (is.character(x)) factor(x) else x)
dat$learning_deficits <- factor(dat$learning_deficits)  # categorical codes, as in the original analysis

message(length(y_vars), " outcomes x ", length(med_vars), " mediators; ", n_cores, " cores")

choose_family_y <- function(x) {
  ux <- sort(unique(na.omit(x)))
  if (length(ux) == 2 && all(ux %in% c(0, 1))) list(name = "binomial", fam = binomial())
  else list(name = "gaussian", fam = gaussian())
}

## -----------------------------
## 3. One mediator-outcome model
## -----------------------------
run_one <- function(y, m, seed) {
  vars_needed <- c(treat_var, y, m, covars)
  df_sub <- dat[complete.cases(dat[, vars_needed]), vars_needed, drop = FALSE]

  if (nrow(df_sub) < 40) return(NULL)
  if (var(df_sub[[treat_var]]) == 0 || var(df_sub[[m]]) == 0) return(NULL)

  fam <- choose_family_y(dat[[y]])
  out_model <- tryCatch(glm(reformulate(c(treat_var, m, covars), y), data = df_sub, family = fam$fam),
                        error = function(e) NULL)
  med_model <- tryCatch(glm(reformulate(c(treat_var, covars), m), data = df_sub, family = gaussian()),
                        error = function(e) NULL)
  if (is.null(out_model) || is.null(med_model)) return(NULL)

  set.seed(seed)
  fit <- tryCatch(
    mediate(model.m = med_model, model.y = out_model, treat = treat_var, mediator = m,
            treat.value = max(df_sub[[treat_var]]), control.value = min(df_sub[[treat_var]]),
            sims = n_sims),
    error = function(e) NULL
  )
  if (is.null(fit)) return(NULL)

  data.frame(
    y = y, mediator = m, n = fit$nobs, family_y = fam$name, seed = seed,
    acme = fit$d.avg, acme_ci_low = fit$d.avg.ci[1], acme_ci_high = fit$d.avg.ci[2], acme_p = fit$d.avg.p,
    ade = fit$z.avg, ade_ci_low = fit$z.avg.ci[1], ade_ci_high = fit$z.avg.ci[2], ade_p = fit$z.avg.p,
    total_effect = fit$tau.coef, total_ci_low = fit$tau.ci[1], total_ci_high = fit$tau.ci[2], total_p = fit$tau.p,
    prop_med = fit$n.avg, prop_med_ci_low = fit$n.avg.ci[1], prop_med_ci_high = fit$n.avg.ci[2], prop_med_p = fit$n.avg.p,
    stringsAsFactors = FALSE
  )
}

## -----------------------------
## 4. Grid, one outcome at a time
## -----------------------------
args    <- as.integer(commandArgs(trailingOnly = TRUE))
y_range <- if (length(args) == 2) args[1]:args[2] else seq_along(y_vars)

for (yi in y_range) {
  y <- y_vars[yi]
  part_file <- file.path(parts_dir, paste0(y, ".csv"))
  if (file.exists(part_file)) { message("[", yi, "/", length(y_vars), "] ", y, ": done, skipping"); next }

  t0 <- Sys.time()
  seeds <- base_seed + (yi - 1L) * length(med_vars) + seq_along(med_vars)
  res <- mclapply(seq_along(med_vars), function(mi) run_one(y, med_vars[mi], seeds[mi]),
                  mc.cores = n_cores)
  failed <- vapply(res, inherits, logical(1), what = "try-error")
  if (any(failed)) stop("Worker error for ", y, ": ", as.character(res[[which(failed)[1]]]))

  part <- bind_rows(res)
  write.csv(part, paste0(part_file, ".tmp"), row.names = FALSE)
  file.rename(paste0(part_file, ".tmp"), part_file)
  message("[", yi, "/", length(y_vars), "] ", y, ": ", nrow(part), " models in ",
          round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), " min")
}

## -----------------------------
## 5. Combine, FDR and save (once every outcome is done)
## -----------------------------
part_files <- file.path(parts_dir, paste0(y_vars, ".csv"))
if (!all(file.exists(part_files))) {
  message(sum(file.exists(part_files)), "/", length(y_vars), " outcomes done; combine runs when all are finished.")
  quit(save = "no")
}

all_res <- bind_rows(lapply(part_files, read.csv, check.names = FALSE))
all_res$acme_p_fdr <- p.adjust(all_res$acme_p, method = "fdr")
sig_res <- filter(all_res, acme_p_fdr < 0.05)

write.csv(all_res, out_all, row.names = FALSE)
write.csv(sig_res, out_sig, row.names = FALSE)
message("Saved ", nrow(all_res), " models to ", out_all)
message("Significant (ACME FDR < 0.05): ", nrow(sig_res), " -> ", out_sig)
