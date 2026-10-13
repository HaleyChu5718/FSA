library(tidyverse)
library(truncnorm)
library(circular)

# FSA Simulation
#
# Generates synthetic FSA data matching the preproc.csv format so it can be
# piped directly into run_fsa_analysis_peak() in fsa_max_peak_updated_with_spectra.R
#
# Response model (per trial):
#   eta_t = baseline + A_i * sin(2pi * f_i * SOA_t + phi_i) + trial_effects
#   p(correct)_t = logistic(eta_t)
#   BRV_t ~ Bernoulli(p) -> +1 or -1, then NA with probability p_nr

logistic <- function(x) 1 / (1 + exp(-x))

# generates a sequence of values where no single value repeats more than max_run times consecutively
constrained_sequence <- function(n, values = c(-1, 1), max_run = 4) {
  seq_out <- numeric(n)
  for (t in seq_len(n)) {
    allowed <- values
    if (t > max_run) {
      if (length(unique(seq_out[(t - max_run):(t - 1)])) == 1) {
        allowed <- setdiff(values, seq_out[t - 1])
      }
    }
    seq_out[t] <- sample(allowed, 1)
  }
  seq_out
}

simulate_subject <- function(
    subID,
    n_trials       = 450,
    # rhythm parameters
    amplitude      = 0.3,     # A: mean rhythm strength (log-odds)
    sd_amp         = 0.1,     # sigma_A: between-subject SD of amplitude
    freq_target    = 8,      # f_target: target frequency (Hz)
    sd_freq        = 0,       # sigma_f: between-subject SD of frequency
    kappa          = 0,       # von Mises concentration (0 = random phase)
    # baseline: per-subject accuracy drawn from Beta(alpha_acc, beta_acc)
    # fitted from 8 real FSA subjects (mean=0.745, SD=0.062)
    alpha_acc      = 37.4,
    beta_acc       = 12.8,
    # trial-level effect SDs (each beta drawn per subject from Normal(0, sigma))
    sd_fatigue     = 0,
    sd_side        = 0,
    sd_tilt        = 0,
    sd_hemi        = 0,
    sd_sequential  = 0,
    # no-response
    p_nr           = 0.05,
    # SOA timing jitter: actual_SOA = intended_SOA + Normal(jitter_mean, jitter_sd)
    # TODO: fill in from real data (run: df$actual_SOA - df$intended_SOA per subject)
    jitter_mean    = NA,   # mean timing error (seconds) — e.g. small positive bias from refresh rounding
    jitter_sd      = NA    # SD of timing error (seconds)
) {

  # --- subject-level parameters ---
  A_i   <- rtruncnorm(1, a = 0, b = Inf, mean = amplitude, sd = sd_amp)
  f_i   <- max(1, min(30, rnorm(1, mean = freq_target, sd = sd_freq)))
  phi_i <- as.numeric(rvonmises(1, mu = circular(0), kappa = kappa))

  beta_fatigue    <- rnorm(1, 0, sd_fatigue)
  beta_side       <- rnorm(1, 0, sd_side)
  beta_tilt       <- rnorm(1, 0, sd_tilt)
  beta_hemi       <- rnorm(1, 0, sd_hemi)
  beta_sequential <- rnorm(1, 0, sd_sequential)

  acc_i            <- rbeta(1, shape1 = alpha_acc, shape2 = beta_acc)
  baseline_logodds <- log(acc_i / (1 - acc_i))

  # --- SOA: stratified uniform across 61 bins (1.0 to 2.0 s at 1/60 s steps) ---
  soa_bins  <- seq(1.0, 2.0, by = 1/60)          # 61 values
  base_reps <- floor(n_trials / length(soa_bins)) # trials per bin
  remainder <- n_trials - base_reps * length(soa_bins)
  intended_SOAs <- c(
    rep(soa_bins, each = base_reps),
    sample(soa_bins, remainder, replace = FALSE)
  )
  intended_SOAs <- sample(intended_SOAs)          # shuffle trial order

  # actual_SOA is a placeholder — set equal to intended_SOA for now.
  # jitter_mean and jitter_sd are reserved for a timing metric to be defined later.
  actual_SOAs <- intended_SOAs

  # --- trial-level covariates ---
  trial_index <- seq_len(n_trials)
  cue_dir     <- constrained_sequence(n_trials, max_run = 4)
  tilt_dir    <- constrained_sequence(n_trials, max_run = 4)

  # --- generate responses ---
  BRV_accuracy <- numeric(n_trials)
  prev_BRV     <- 0   # no previous trial at start

  for (t in seq_len(n_trials)) {

    rhythm_component <- A_i * sin(2 * pi * f_i * actual_SOAs[t] + phi_i)

    trial_effects <-
      beta_fatigue    * (trial_index[t] / n_trials) +
      beta_side       * cue_dir[t] +
      beta_tilt       * tilt_dir[t] +
      beta_hemi       * (cue_dir[t] * tilt_dir[t]) +
      beta_sequential * prev_BRV

    eta   <- baseline_logodds + rhythm_component + trial_effects
    p_cor <- logistic(eta)

    response      <- ifelse(runif(1) < p_cor, 1, -1)
    BRV_accuracy[t] <- ifelse(runif(1) < p_nr, NA, response)
    prev_BRV      <- ifelse(is.na(BRV_accuracy[t]), 0, BRV_accuracy[t])
  }

  is_correct <- ifelse(is.na(BRV_accuracy), NA, BRV_accuracy == 1)
  responded  <- !is.na(BRV_accuracy)

  tibble(
    subID        = subID,
    block        = 1L,
    trials.thisN = trial_index - 1L,
    intended_SOA = intended_SOAs,
    actual_SOA   = actual_SOAs,
    BRV_accuracy = BRV_accuracy,
    is_correct   = is_correct,
    responded    = responded,
    cue_dir      = cue_dir,
    tilt_dir     = tilt_dir,
    # store true parameters as metadata columns (dropped before analysis if needed)
    true_A       = A_i,
    true_freq    = f_i,
    true_phase   = phi_i
  )
}

# MAIN SIMULATION WRAPPER

simulate_fsa <- function(
    n_subjects     = 40,
    n_trials       = 450,
    # rhythm
    amplitude      = 0.3,
    sd_amp         = 0.1,
    freq_target    = 8,
    sd_freq        = 0,
    kappa          = 0,       # phase consistency: 0 = random, >0 = aligned
    # baseline: per-subject accuracy drawn from Beta(alpha_acc, beta_acc)
    # fitted from 8 real FSA subjects (mean=0.745, SD=0.062); update via fsa_fit_beta_accuracy.R
    alpha_acc      = 37.4,
    beta_acc       = 12.8,
    # trial-level effect SDs
    sd_fatigue     = 0,
    sd_side        = 0,
    sd_tilt        = 0,
    sd_hemi        = 0,
    sd_sequential  = 0,
    # no-response
    p_nr           = 0.05,
    # SOA timing jitter — TODO: fill in from real data
    jitter_mean    = NA,
    jitter_sd      = NA,
    # output
    out_root       = "~/Desktop/FSA/simulated",
    seed           = 42
) {

  set.seed(seed)

  out_root <- path.expand(out_root)

  message(sprintf(
    "Simulating %d subjects x %d trials | A=%.2f sd_A=%.2f | f=%.1f Hz sd_f=%.2f | kappa=%.1f | p_nr=%.2f",
    n_subjects, n_trials, amplitude, sd_amp, freq_target, sd_freq, kappa, p_nr
  ))

  all_data <- map_dfr(seq_len(n_subjects), function(i) {
    subID <- sprintf("SIM%03d", i)
    simulate_subject(
      subID          = subID,
      n_trials       = n_trials,
      amplitude      = amplitude,
      sd_amp         = sd_amp,
      freq_target    = freq_target,
      sd_freq        = sd_freq,
      kappa          = kappa,
      alpha_acc      = alpha_acc,
      beta_acc       = beta_acc,
      sd_fatigue     = sd_fatigue,
      sd_side        = sd_side,
      sd_tilt        = sd_tilt,
      sd_hemi        = sd_hemi,
      sd_sequential  = sd_sequential,
      p_nr           = p_nr,
      jitter_mean    = jitter_mean,
      jitter_sd      = jitter_sd
    )
  })

  # save one preproc.csv per subject (matches real data structure)
  analysis_cols <- c("block", "trials.thisN", "intended_SOA", "actual_SOA",
                     "BRV_accuracy", "is_correct", "responded", "cue_dir", "tilt_dir")

  walk(unique(all_data$subID), function(sid) {
    sub_dir <- file.path(out_root, sid)
    if (!dir.exists(sub_dir)) dir.create(sub_dir, recursive = TRUE)
    all_data %>%
      filter(subID == sid) %>%
      select(all_of(analysis_cols)) %>%
      write.csv(file.path(sub_dir, "preproc.csv"), row.names = FALSE)
  })

  # summary
  summary_df <- all_data %>%
    group_by(subID) %>%
    summarise(
      true_A     = first(true_A),
      true_freq  = first(true_freq),
      true_phase = first(true_phase),
      n_trials   = n(),
      pct_responded = mean(responded) * 100,
      accuracy   = mean(is_correct, na.rm = TRUE) * 100,
      .groups    = "drop"
    )

  cat("\n=== SIMULATION SUMMARY ===\n")
  cat(sprintf("Output saved to: %s\n", out_root))
  cat(sprintf("Mean amplitude:  %.3f (SD %.3f)\n",
              mean(summary_df$true_A), sd(summary_df$true_A)))
  cat(sprintf("Mean frequency:  %.2f Hz (SD %.2f)\n",
              mean(summary_df$true_freq), sd(summary_df$true_freq)))
  cat(sprintf("Mean accuracy:   %.1f%%\n", mean(summary_df$accuracy)))
  cat(sprintf("Mean response rate: %.1f%%\n", mean(summary_df$pct_responded)))

  invisible(list(all_data = all_data, summary = summary_df))
}

# --- Example calls ---

# Null simulation (no rhythm — tests false positive rate)
# simulate_fsa(amplitude = 0, sd_amp = 0, out_root = "~/Desktop/FSA/simulated/null")

# Single frequency, aligned phases, moderate SNR
# simulate_fsa(amplitude = 0.3, freq_target = 10, kappa = 2,
#              out_root = "~/Desktop/FSA/simulated/signal_aligned")

# Single frequency, random phases (tests phase-agnostic detection)
# simulate_fsa(amplitude = 0.3, freq_target = 10, kappa = 0,
#              out_root = "~/Desktop/FSA/simulated/signal_random_phase")

# With between-subject frequency variability
# simulate_fsa(amplitude = 0.3, freq_target = 10, sd_freq = 2,
#              out_root = "~/Desktop/FSA/simulated/signal_freq_variability")

# With trial-level confounds
# simulate_fsa(amplitude = 0.3, freq_target = 10,
#              sd_fatigue = 0.2, sd_sequential = 0.15,
#              out_root = "~/Desktop/FSA/simulated/signal_with_confounds")

simulate_fsa()
