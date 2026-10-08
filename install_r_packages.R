# R 4.3.1. Installs the package versions used for the mediation analyses.
# Usage: Rscript install_r_packages.R
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_version("mediation", version = "4.5.1", upgrade = "never")
remotes::install_version("tidyverse", version = "2.0.0", upgrade = "never")
