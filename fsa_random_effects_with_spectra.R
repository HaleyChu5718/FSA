library(tidyverse)

# Raw data source: ~/Library/CloudStorage/GoogleDrive-haleychu@stanford.edu/My Drive/FSA Data
# Run fsa_preprocessing.R first to generate preprocessed/ from raw data.
#
# Implements the random-effects (RE) test for existence of behavioral rhythmicity
# following Tosato et al. (2022) NeuroImage, doi:10.1016/j.neuroimage.2022.119561
#
# Spectral method: stLSS (single-trial least-squares spectrum, Section 2.4.3)
#   Per subject, per frequency: lm(BRV ~ sin + cos) → c_f = β1 + iβ2
#
# Group statistic (PAWAAS, Section 2.8):
#   |mean(c_f across subjects)|^2 — phase-aligned within AND across subjects
#
# Null / bias estimate (Section 2.5):
#   Per subject: shuffle BRV_accuracy within subject n_bias_perms times,
#   average the resulting complex spectra → one bias-estimate complex spectrum.
#   Per-subject bias power = |bias_complex_i|^2.
#
# Statistical test (RE, Section 2.5):
#   Paired one-sided t-test across subjects:
#     H₀: mean(|observed_i|^2 − |bias_i|^2) = 0
#   Max-based multiple comparison correction:
#     For each of n_perms permutations, randomly flip the sign of the difference
#     per subject (equivalent to randomly exchanging observed/bias), recompute
#     t-statistic per frequency, record the maximum across frequencies.
#     Threshold = 95th percentile of this max distribution.

# STEP 1: stLSS — complex spectrum per subject
# Returns a complex vector, one value per frequency
compute_complex_spectrum <- function(df_subject, freq_range = 4:18) {
  sapply(freq_range, function(freq) {
    fit <- lm(BRV_accuracy ~ sin(2*pi*freq*actual_SOA) +
                cos(2*pi*freq*actual_SOA),
              data = df_subject)
    complex(real = coef(fit)[2], imaginary = coef(fit)[3])
  })
}

# STEP 2: per-subject bias-estimate spectrum
# Shuffles BRV_accuracy within subject n_bias_perms times.
# For each shuffle: compute complex spectrum → take power (|c_f|^2).
# Returns the average of those n_bias_perms power spectra (real-valued vector).
# This gives the expected noise-floor power per subject under H0.
# NB: averaging power spectra (not complex spectra) is correct here because the
# t-test in Step 3 requires a real-valued per-subject scalar per frequency.
# Averaging complex spectra across shuffles would yield ~0 (random phases cancel),
# making the bias meaningless.
compute_bias_spectrum <- function(df_subject, freq_range = 4:18, n_bias_perms = 1000) {
  valid_df  <- df_subject %>% filter(!is.na(BRV_accuracy))
  power_sum <- numeric(length(freq_range))
  for (p in seq_len(n_bias_perms)) {
    shuffled  <- valid_df %>% mutate(BRV_accuracy = sample(BRV_accuracy))
    power_sum <- power_sum + Mod(compute_complex_spectrum(shuffled, freq_range))^2
  }
  power_sum / n_bias_perms  # average power spectrum (real-valued) per subject
}

# STEP 3: RE t-test with max-based multiple comparison correction
# observed_power_mat, bias_power_mat: n_subjects x n_freq matrices of power (|complex|^2)
# For each permutation, randomly flip the sign of (observed - bias) per subject,
# recompute t-statistic, record the max → 95th percentile = corrected threshold.
compute_re_significance <- function(observed_power_mat, bias_power_mat,
                                    freq_range = 4:18, n_perms = 2000, alpha = 0.05) {
  diff_mat   <- observed_power_mat - bias_power_mat  # n_subjects x n_freq
  n_subjects <- nrow(diff_mat)

  # observed one-sided t-statistic and p-value per frequency
  t_obs <- apply(diff_mat, 2, function(d)
    t.test(d, mu = 0, alternative = "greater")$statistic)
  p_obs <- apply(diff_mat, 2, function(d)
    t.test(d, mu = 0, alternative = "greater")$p.value)

  # max-based permutation null: flip sign of difference per subject
  t_null_max <- numeric(n_perms)
  for (perm in seq_len(n_perms)) {
    flip      <- sample(c(-1, 1), n_subjects, replace = TRUE)
    diff_perm <- sweep(diff_mat, 1, flip, "*")
    t_perm    <- apply(diff_perm, 2, function(d)
      t.test(d, mu = 0, alternative = "greater")$statistic)
    t_null_max[perm] <- max(t_perm)
  }

  threshold        <- quantile(t_null_max, 1 - alpha)
  significant_corr <- t_obs > threshold

  data.frame(
    freq             = freq_range,
    t_statistic      = t_obs,
    p_value          = p_obs,
    significant_corr = significant_corr,
    t_threshold      = threshold
  )
}

# MAIN WRAPPER FUNCTION

run_fsa_random_effects_with_spectra <- function(
    preproc_root  = "~/Desktop/FSA/preprocessed",
    results_root  = "~/Desktop/FSA/results_random_effects_with_spectra",
    freq_range    = 4:18,
    n_bias_perms  = 1000,   # within-subject shuffles for bias estimate (Section 2.5)
    n_perms       = 2000,   # permutations for max-based RE t-test (Section 2.5)
    alpha         = 0.05,
    sub_ids       = NULL,
    seed          = 42
) {

  set.seed(seed)

  # 1. load preprocessed data
  message("Loading preprocessed data...")
  sub_folders <- list.dirs(path.expand(preproc_root), full.names = TRUE, recursive = FALSE)
  if (!is.null(sub_ids)) sub_folders <- sub_folders[basename(sub_folders) %in% sub_ids]

  df <- map_dfr(sub_folders, function(folder) {
    subID <- basename(folder)
    data  <- read.csv(file.path(folder, "preproc.csv"))
    data$subID <- subID
    return(data)
  })
  message(sprintf("Loaded %d trials from %d subjects", nrow(df), length(unique(df$subID))))
  subIDs <- unique(df$subID)

  # 2. compute observed complex spectrum per subject (stLSS)
  message("Computing observed spectra (stLSS)...")
  subject_complex <- lapply(subIDs, function(sid) {
    compute_complex_spectrum(filter(df, subID == sid, !is.na(BRV_accuracy)), freq_range)
  })
  names(subject_complex) <- subIDs

  # per-subject observed power: |c_f_i|^2
  observed_power_mat <- do.call(rbind, lapply(subject_complex, function(c) Mod(c)^2))
  rownames(observed_power_mat) <- subIDs

  # group observed power (PAWAAS): |mean(complex across subjects)|^2
  avg_complex <- Reduce("+", subject_complex) / length(subject_complex)
  group_power <- Mod(avg_complex)^2

  # 3. compute bias-estimate spectrum per subject (1000 within-subject shuffles, averaged)
  message(sprintf("Computing bias-estimate spectra per subject (%d shuffles each)...", n_bias_perms))
  bias_spectra <- lapply(subIDs, function(sid) {
    message(sprintf("  bias: %s", sid))
    compute_bias_spectrum(filter(df, subID == sid), freq_range, n_bias_perms)
  })
  names(bias_spectra) <- subIDs

  # per-subject bias power: already real-valued (average of |c_f_shuffle|^2 over shuffles)
  bias_power_mat <- do.call(rbind, lapply(bias_spectra, function(x) x))
  rownames(bias_power_mat) <- subIDs

  # 4. RE paired t-test with max-based correction
  message(sprintf("Running random-effects t-test with max-based correction (%d permutations)...", n_perms))
  results_df <- compute_re_significance(observed_power_mat, bias_power_mat,
                                        freq_range = freq_range,
                                        n_perms    = n_perms,
                                        alpha      = alpha)

  cat("\n=== RESULTS ===\n")
  cat(sprintf("N subjects: %d\n", length(subIDs)))
  cat(sprintf("Significant frequencies (uncorrected p<%.2f): %s\n",
              alpha, paste(results_df$freq[results_df$p_value < alpha], collapse = ", ")))
  cat(sprintf("Significant frequencies (max-corrected RE t-test): %s\n",
              paste(results_df$freq[results_df$significant_corr], collapse = ", ")))
  cat(sprintf("Max-corrected t-threshold (alpha=%.2f): %.3f\n", alpha, results_df$t_threshold[1]))

  # 5. build group spectra data frame for plotting
  mean_obs_power  <- colMeans(observed_power_mat)
  sd_obs          <- apply(observed_power_mat, 2, sd)
  sem_obs         <- sd_obs / sqrt(nrow(observed_power_mat))

  mean_bias_power <- colMeans(bias_power_mat)
  sd_bias         <- apply(bias_power_mat, 2, sd)
  sem_bias        <- sd_bias / sqrt(nrow(bias_power_mat))

  group_spectra <- tibble(
    freq       = rep(freq_range, 2),
    type       = rep(c("Empirical", "Null (bias avg)"), each = length(freq_range)),
    mean_power = c(mean_obs_power, mean_bias_power),
    sd_power   = c(sd_obs, sd_bias),
    sem        = c(sem_obs, sem_bias)
  ) %>%
    mutate(
      lower_sem = mean_power - sem,
      upper_sem = mean_power + sem,
      lower_sd  = mean_power - sd_power,
      upper_sd  = mean_power + sd_power
    )

  per_subject <- bind_rows(
    map_dfr(subIDs, function(sid) tibble(
      subID = sid, freq = freq_range,
      power = observed_power_mat[sid, ], type = "Empirical"
    )),
    map_dfr(subIDs, function(sid) tibble(
      subID = sid, freq = freq_range,
      power = bias_power_mat[sid, ], type = "Null (bias avg)"
    ))
  )

  # 6. save CSVs
  if (!dir.exists(path.expand(results_root))) dir.create(path.expand(results_root), recursive = TRUE)
  write.csv(per_subject,   file.path(path.expand(results_root), "per_subject_spectra.csv"),  row.names = FALSE)
  write.csv(group_spectra, file.path(path.expand(results_root), "group_spectra.csv"),        row.names = FALSE)
  write.csv(results_df,    file.path(path.expand(results_root), "significance_results.csv"), row.names = FALSE)

  # significance annotation: frequencies passing max-corrected threshold
  sig_freqs  <- results_df %>% filter(significant_corr)
  y_star     <- max(group_spectra$mean_power) * 1.08

  # PLOT 1: group mean empirical vs bias with SEM ribbon
  p1 <- ggplot(group_spectra,
               aes(x = freq, y = mean_power, color = type, fill = type)) +
    geom_ribbon(aes(ymin = lower_sem, ymax = upper_sem),
                alpha = 0.15, color = NA) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +
    { if (nrow(sig_freqs) > 0)
        geom_point(data = sig_freqs, aes(x = freq, y = y_star),
                   inherit.aes = FALSE, shape = 8, size = 3, color = "black")
    } +
    scale_x_continuous(breaks = freq_range) +
    scale_color_manual(values = c("Empirical" = "#CC79A7", "Null (bias avg)" = "#88CCEE")) +
    scale_fill_manual(values  = c("Empirical" = "#CC79A7", "Null (bias avg)" = "#88CCEE")) +
    labs(
      title    = "Mean Empirical vs. Bias-Estimate Power Spectrum (Random Effects)",
      subtitle = sprintf(
        "N = %d subjects | bias = mean of %d within-subject shuffles | shaded = ±1 SEM | * = max-corrected RE t-test p<%.2f",
        length(subIDs), n_bias_perms, alpha),
      x = "Frequency (Hz)", y = "Power", color = NULL, fill = NULL
    ) +
    theme_minimal() +
    theme(legend.position = "top")

  ggsave(file.path(path.expand(results_root), "mean_empirical_vs_null_spectra.png"),
         p1, width = 9, height = 5.5)
  print(p1)

  # PLOT 2: individual subject lines + SD ribbon + group mean, faceted
  p2 <- ggplot(mapping = aes(x = freq, color = type, fill = type)) +
    geom_line(data = per_subject,
              aes(y = power, group = subID),
              alpha = 0.2, linewidth = 0.35) +
    geom_ribbon(data = group_spectra,
                aes(y = mean_power, ymin = lower_sd, ymax = upper_sd),
                alpha = 0.2, color = NA) +
    geom_line(data = group_spectra,
              aes(y = mean_power), linewidth = 1.1) +
    geom_point(data = group_spectra,
               aes(y = mean_power), size = 1.6) +
    scale_x_continuous(breaks = freq_range) +
    facet_wrap(~ type) +
    scale_color_manual(values = c("Empirical" = "#CC79A7", "Null (bias avg)" = "#88CCEE")) +
    scale_fill_manual(values  = c("Empirical" = "#CC79A7", "Null (bias avg)" = "#88CCEE")) +
    labs(
      title    = "Empirical vs. Bias-Estimate Power Spectrum with Individual Subjects",
      subtitle = sprintf(
        "N = %d subjects | thin lines = individual subjects | shaded = ±1 SD | bold = group mean",
        length(subIDs)),
      x = "Frequency (Hz)", y = "Power", color = NULL, fill = NULL
    ) +
    theme_minimal() +
    theme(legend.position = "none")

  ggsave(file.path(path.expand(results_root), "empirical_vs_null_spectra_with_individuals.png"),
         p2, width = 11, height = 6)
  print(p2)

  message(sprintf("Results saved to %s", results_root))

  invisible(list(
    df                 = df,
    subject_complex    = subject_complex,
    observed_power_mat = observed_power_mat,
    bias_spectra       = bias_spectra,
    bias_power_mat     = bias_power_mat,
    group_power        = group_power,
    per_subject        = per_subject,
    group_spectra      = group_spectra,
    results_df         = results_df
  ))
}

# run_fsa_random_effects_with_spectra()
