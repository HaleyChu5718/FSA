library(tidyverse)

# Raw data source: ~/Library/CloudStorage/GoogleDrive-haleychu@stanford.edu/My Drive/FSA Data
# Run fsa_preprocessing.R first to generate preprocessed/ from raw data.
#
# Identical analysis to fsa_max_peak_updated.R with one addition: every
# split's half-spectra are accumulated into a running per-frequency sum/count
# (NA-safe) for both empirical and null computations. This lets us plot the
# mean empirical vs. null spectrum actually averaged across the 1000 splits
# (and 20 surrogates) that produced z_obs/z_null/delta_s.

# STEP 1: observed peak-agreement spectrum
# Returns a list: replication_rate, spectrum_sum, spectrum_count
compute_z_obs_peak <- function(df_subject,
                               n_splits       = 1000,
                               freq_range     = 4:18,
                               peak_threshold = 1.33,
                               detrend        = TRUE) {

  compute_spectrum <- function(half_df) {
    sapply(freq_range, function(freq) {
      fit <- lm(BRV_accuracy ~ sin(2*pi*freq*actual_SOA) +
                  cos(2*pi*freq*actual_SOA),
                data = half_df)
      power <- sum(coef(fit)[2:3]^2)
      if (!is.finite(power)) return(NA)
      power
    })
  }

  # trial-axis detrend once before the split loop
  df_ready <- df_subject
  if (detrend && length(unique(df_subject$block)) >= 2) {
    detrend_model <- lm(BRV_accuracy ~ trials.thisN * block,
                        data = df_subject, na.action = na.exclude)
    df_ready <- df_subject %>%
      mutate(BRV_accuracy = residuals(detrend_model))
  }

  peak_agree_vals <- numeric(n_splits)

  n_freq         <- length(freq_range)
  spectrum_sum   <- numeric(n_freq)
  spectrum_count <- numeric(n_freq)

  for (s in seq_len(n_splits)) {

    split_trials <- df_ready %>%
      filter(!is.na(BRV_accuracy)) %>%
      group_by(intended_SOA) %>%
      mutate(
        trial_order = sample(n()),
        half        = ifelse(trial_order <= floor(n()/2), 1, 2)
      ) %>%
      ungroup()

    spec1 <- compute_spectrum(filter(split_trials, half == 1))
    spec2 <- compute_spectrum(filter(split_trials, half == 2))

    # accumulate per-frequency sums (NA-safe)
    valid1 <- !is.na(spec1)
    valid2 <- !is.na(spec2)
    spectrum_sum[valid1]   <- spectrum_sum[valid1] + spec1[valid1]
    spectrum_count[valid1] <- spectrum_count[valid1] + 1
    spectrum_sum[valid2]   <- spectrum_sum[valid2] + spec2[valid2]
    spectrum_count[valid2] <- spectrum_count[valid2] + 1

    peak_freq1 <- freq_range[which.max(spec1)]
    peak_freq2 <- freq_range[which.max(spec2)]

    peak_agree_vals[s] <- ifelse(abs(peak_freq1 - peak_freq2) <= peak_threshold, 1, 0)
  }

  list(
    replication_rate = mean(peak_agree_vals),
    spectrum_sum     = spectrum_sum,
    spectrum_count   = spectrum_count
  )
}

# STEP 2: null peak-agreement spectrum (sub-band surrogate)
# Accumulates null spectra across all n_surrogates surrogate datasets
compute_z_null_peak <- function(df_subject,
                                n_surrogates   = 20,
                                trend_freqs    = 1:3,
                                n_splits       = 1000,
                                freq_range     = 4:18,
                                peak_threshold = 1.33) {

  trend_terms <- unlist(lapply(trend_freqs, function(f) {
    c(paste0("sin(2*pi*", f, "*actual_SOA)"),
      paste0("cos(2*pi*", f, "*actual_SOA)"))
  }))
  trend_formula <- as.formula(paste("BRV_accuracy ~",
                                    paste(trend_terms, collapse = " + ")))
  trend_model <- lm(trend_formula, data = df_subject, na.action = na.exclude)
  f_hat <- fitted(trend_model)
  resid  <- residuals(trend_model)

  null_agree_vals <- numeric(n_surrogates)

  n_freq              <- length(freq_range)
  null_spectrum_sum   <- numeric(n_freq)
  null_spectrum_count <- numeric(n_freq)

  for (i in seq_len(n_surrogates)) {
    shuffled_resid <- sample(resid)
    BRV_surrogate  <- f_hat + shuffled_resid
    df_surrogate   <- df_subject %>%
      mutate(BRV_accuracy = BRV_surrogate)
    obs_result <- suppressWarnings(
      compute_z_obs_peak(df_surrogate,
                         n_splits       = n_splits,
                         freq_range     = freq_range,
                         peak_threshold = peak_threshold,
                         detrend        = FALSE)
    )
    null_agree_vals[i]  <- obs_result$replication_rate
    null_spectrum_sum   <- null_spectrum_sum   + obs_result$spectrum_sum
    null_spectrum_count <- null_spectrum_count + obs_result$spectrum_count
  }

  list(
    replication_rate = mean(null_agree_vals),
    spectrum_sum     = null_spectrum_sum,
    spectrum_count   = null_spectrum_count
  )
}

# STEP 3: per-subject delta (kept for standalone use)
compute_delta_peak <- function(df_subject,
                               n_splits       = 1000,
                               n_surrogates   = 20,
                               freq_range     = 4:18,
                               trend_freqs    = 1:3,
                               peak_threshold = 1.33) {
  z_obs_result  <- compute_z_obs_peak(df_subject,
                                      n_splits       = n_splits,
                                      freq_range     = freq_range,
                                      peak_threshold = peak_threshold)
  z_null_result <- compute_z_null_peak(df_subject,
                                       n_surrogates   = n_surrogates,
                                       trend_freqs    = trend_freqs,
                                       n_splits       = n_splits,
                                       freq_range     = freq_range,
                                       peak_threshold = peak_threshold)
  z_obs_result$replication_rate - z_null_result$replication_rate
}

# MAIN WRAPPER FUNCTION

run_fsa_analysis_peak <- function(
    preproc_root   = "~/Desktop/FSA/preprocessed",
    results_root   = "~/Desktop/FSA/results_max_peak_updated_with_spectra",
    n_splits       = 1000,
    n_surrogates   = 20,
    freq_range     = 4:18,
    trend_freqs    = 1:3,
    peak_threshold = 1.33,
    sub_ids        = NULL
) {

  set.seed(42)

  # 1. load preprocessed data
  message("Loading preprocessed data...")
  sub_folders <- list.dirs(preproc_root, full.names = TRUE, recursive = FALSE)

  if (!is.null(sub_ids)) {
    sub_folders <- sub_folders[basename(sub_folders) %in% sub_ids]
  }

  df <- map_dfr(sub_folders, function(folder) {
    subID <- basename(folder)
    data  <- read.csv(file.path(folder, "preproc.csv"))
    data$subID <- subID
    return(data)
  })
  message(sprintf("Loaded %d trials from %d subjects", nrow(df), length(unique(df$subID))))

  subIDs <- unique(df$subID)

  # 2. compute z_obs per subject (also accumulates empirical spectra)
  message("Computing z_obs_peak for all subjects...")
  z_obs_results <- lapply(subIDs, function(sid) {
    message(sprintf("  z_obs: %s", sid))
    compute_z_obs_peak(filter(df, subID == sid),
                       n_splits       = n_splits,
                       freq_range     = freq_range,
                       peak_threshold = peak_threshold)
  })
  names(z_obs_results) <- subIDs
  z_obs_all_peak <- sapply(z_obs_results, function(r) r$replication_rate)

  # 3. compute z_null per subject (also accumulates null spectra)
  message("Computing z_null_peak for all subjects...")
  z_null_results <- lapply(subIDs, function(sid) {
    message(sprintf("  z_null: %s", sid))
    compute_z_null_peak(filter(df, subID == sid),
                        n_surrogates   = n_surrogates,
                        trend_freqs    = trend_freqs,
                        n_splits       = n_splits,
                        freq_range     = freq_range,
                        peak_threshold = peak_threshold)
  })
  names(z_null_results) <- subIDs
  z_null_all_peak <- sapply(z_null_results, function(r) r$replication_rate)

  # 4. compute delta per subject
  message("Computing delta_s...")
  delta_all_peak <- z_obs_all_peak - z_null_all_peak

  # 5. group inference (Wilcoxon — BRV_accuracy is binary)
  message("Running group inference...")
  if (length(delta_all_peak) >= 3) {
    shapiro_result <- shapiro.test(delta_all_peak)
  } else {
    shapiro_result <- list(p.value = NA, statistic = NA)
    message("Shapiro-Wilk skipped: need at least 3 subjects")
  }
  t_result <- t.test(delta_all_peak, mu = 0, alternative = "greater")
  w_result <- wilcox.test(delta_all_peak, mu = 0, alternative = "greater")

  cat("\n=== RESULTS ===\n")
  cat(sprintf("N subjects: %d\n", length(delta_all_peak)))
  cat(sprintf("Mean delta_s: %.4f\n", mean(delta_all_peak)))
  cat(sprintf("Shapiro-Wilk p: %.4f\n", shapiro_result$p.value))
  cat(sprintf("t-test p: %.4f\n", t_result$p.value))
  cat(sprintf("Wilcoxon p: %.4f\n", w_result$p.value))

  # normality plots
  hist(delta_all_peak, main = "Distribution of delta_s (peak)", xlab = "delta_s", breaks = 10)
  qqnorm(delta_all_peak); qqline(delta_all_peak)

  # 6. save main results
  if (!dir.exists(results_root)) dir.create(results_root, recursive = TRUE)
  results_df <- data.frame(
    subID   = names(delta_all_peak),
    z_obs   = z_obs_all_peak,
    z_null  = z_null_all_peak,
    delta_s = delta_all_peak
  )
  write.csv(results_df,
            file.path(results_root, "rhythmic_analysis_results_peak.csv"),
            row.names = FALSE)
  message(sprintf("Results saved to %s", results_root))

  # 7. build per-subject mean spectra from accumulated sums/counts
  message("Building mean spectra across all splits/surrogates...")
  emp_spectra <- map_dfr(subIDs, function(sid) {
    r <- z_obs_results[[sid]]
    tibble(subID = sid, freq = freq_range,
           power = r$spectrum_sum / r$spectrum_count,
           type  = "Empirical")
  })
  null_spectra <- map_dfr(subIDs, function(sid) {
    r <- z_null_results[[sid]]
    tibble(subID = sid, freq = freq_range,
           power = r$spectrum_sum / r$spectrum_count,
           type  = "Null (surrogate avg)")
  })
  all_spectra <- bind_rows(emp_spectra, null_spectra)

  group_spectra <- all_spectra %>%
    group_by(type, freq) %>%
    summarise(
      mean_power = mean(power, na.rm = TRUE),
      sd_power   = sd(power,   na.rm = TRUE),
      sem        = sd_power / sqrt(sum(!is.na(power))),
      .groups    = "drop"
    ) %>%
    mutate(lower = mean_power - sd_power, upper = mean_power + sd_power)

  write.csv(all_spectra,   file.path(results_root, "all_subject_spectra_from_splits.csv"),  row.names = FALSE)
  write.csv(group_spectra, file.path(results_root, "group_mean_spectra_from_splits.csv"),   row.names = FALSE)

  # PLOT 1: group mean empirical vs. null with SEM ribbon
  p_summary <- ggplot(group_spectra,
                      aes(x = freq, y = mean_power, color = type, fill = type)) +
    geom_ribbon(aes(ymin = mean_power - sem, ymax = mean_power + sem),
                alpha = 0.15, color = NA) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +
    scale_x_continuous(breaks = freq_range) +
    labs(
      title    = "Mean Empirical vs. Null Power Spectrum, Averaged Across Actual Splits/Surrogates",
      subtitle = sprintf("N = %d subjects; spectra averaged over %d splits (x2 halves) / %d surrogates; shaded = ±1 SEM across subjects",
                         length(subIDs), n_splits, n_surrogates),
      x = "Frequency (Hz)", y = "Power", color = NULL, fill = NULL
    ) +
    scale_color_manual(values = c("Empirical" = "#CC79A7", "Null (surrogate avg)" = "#88CCEE")) +
    scale_fill_manual(values  = c("Empirical" = "#CC79A7", "Null (surrogate avg)" = "#88CCEE")) +
    theme_minimal() +
    theme(legend.position = "top")

  ggsave(file.path(results_root, "mean_empirical_vs_null_spectra_from_splits.png"),
         p_summary, width = 9, height = 5.5)
  print(p_summary)

  # PLOT 2: individual subject lines + SD ribbon + group mean
  p_individual <- ggplot(mapping = aes(x = freq, color = type, fill = type)) +
    geom_line(data = all_spectra,
              aes(y = power, group = subID),
              alpha = 0.2, linewidth = 0.35) +
    geom_ribbon(data = group_spectra,
                aes(y = mean_power, ymin = lower, ymax = upper),
                alpha = 0.2, color = NA) +
    geom_line(data = group_spectra,
              aes(y = mean_power), linewidth = 1.1) +
    geom_point(data = group_spectra,
               aes(y = mean_power), size = 1.6) +
    scale_x_continuous(breaks = freq_range) +
    facet_wrap(~ type) +
    labs(
      title    = "Mean Empirical vs. Null Power Spectrum, with Individual Subjects",
      subtitle = sprintf("N = %d subjects; spectra averaged over %d splits (x2 halves) / %d surrogates.\nThin lines = individual subjects; shaded band = ±1 SD across subjects; bold line = group mean.",
                         length(subIDs), n_splits, n_surrogates),
      x = "Frequency (Hz)", y = "Power", color = NULL, fill = NULL
    ) +
    scale_color_manual(values = c("Empirical" = "#CC79A7", "Null (surrogate avg)" = "#88CCEE")) +
    scale_fill_manual(values  = c("Empirical" = "#CC79A7", "Null (surrogate avg)" = "#88CCEE")) +
    theme_minimal() +
    theme(legend.position = "none")

  ggsave(file.path(results_root, "mean_empirical_vs_null_spectra_with_individuals.png"),
         p_individual, width = 11, height = 6)
  print(p_individual)

  return(list(
    df            = df,
    z_obs_all     = z_obs_all_peak,
    z_null_all    = z_null_all_peak,
    delta_all     = delta_all_peak,
    t_result      = t_result,
    w_result      = w_result,
    shapiro       = shapiro_result,
    all_spectra   = all_spectra,
    group_spectra = group_spectra
  ))
}

# results <- run_fsa_analysis_peak()
