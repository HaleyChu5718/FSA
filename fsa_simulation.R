library(tidyverse)
library(truncnorm)

# FSA Simulation
#
# Generates synthetic FSA data matching the preproc.csv format so it can be
# piped directly into run_fsa_analysis_peak() in fsa_max_peak_updated_with_spectra.R
#
# Response model (per trial):
#   eta_t = baseline + A_i * sin(2pi * f_i * SOA_t + phi_i) + trial_effects
#   p(correct)_t = logistic(eta_t)
#   BRV_t ~ Bernoulli(p) -> +1 or -1, then NA with probability p_nr

logistic <- function(x) 1 / (1 + exp(-x)) #logistic function for probability of accuracy, converts log-odds value into probability between 0 and 1

# generates a sequence of values where no single value repeats more than max_run times consecutively
constrained_sequence <- function(n, values = c(-1, 1), max_run = 4) {
  seq_out <- numeric(n) #produce numeric array of zeroes that get filled in one at a time
  for (t in seq_len(n)) { #loops through each trial position from 1 to n
    allowed <- values #in the beginning, both 1 and -1 are allowed
    if (t > max_run) { #check for run-length violations once the 4th trial has occurred
      if (length(unique(seq_out[(t - max_run):(t - 1)])) == 1) { #look at previous 4 trials and return the distinct values in that window - if length = 1 meaning only 1 unique value, all 4 previous trials had the same value
        allowed <- setdiff(values, seq_out[t - 1]) #remove the value that's been repeating and force a switch
      }
    }
    seq_out[t] <- sample(allowed, 1) #randomly pick a value or insert the constrained value for the next number
  }
  seq_out
}

simulate_subject <- function(
    subID,
    n_trials       = 450,
    n_blocks       = 6,
    # rhythm parameters
    snr            = 1.0,     # mean signal-to-noise ratio across subjects
    sd_snr         = 0.2,     # between-subject SD of SNR; drawn from TruncatedNormal(snr, sd_snr, lower=0)
    sigma_noise    = 0.3,     # mean noise level (SD of Gaussian white noise added per trial, in log-odds)
    sd_noise       = 0.05,    # between-subject SD of noise level; drawn from TruncatedNormal(sigma_noise, sd_noise, lower=0)
    freq_target    = 8,       # f_target: target frequency (Hz)
    sd_freq        = 0,       # sigma_f: between-subject SD of frequency
    phase_aligned  = FALSE,   # FALSE = uniform random phase; TRUE = all subjects phase = 0
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
    # SOA timing jitter: actual_SOA = intended_SOA + value sampled from observed photodiode distribution
    # empirical values loaded from ~/Desktop/FSA/photodiode_jitter.csv (N=540)
    jitter_values  = NULL    # pass the loaded jitter vector here; NULL disables jitter
) {

  # --- subject-level parameters ---
  sigma_noise_i <- rtruncnorm(1, a = 0, b = Inf, mean = sigma_noise, sd = sd_noise)  # subject noise level
  snr_i         <- rtruncnorm(1, a = 0, b = Inf, mean = snr,         sd = sd_snr)    # subject SNR
  A_i           <- snr_i * sigma_noise_i                                              # amplitude derived from SNR and noise
  f_i   <- max(1, min(30, rnorm(1, mean = freq_target, sd = sd_freq))) #draw subject rhythm frequency from normal distribution with mean = target frequency, max = Nyquist (30Hz), min = 1 Hz
  phi_i <- if (phase_aligned) 0 else runif(1, 0, 2 * pi) #draw subject phase from von Mises distribution centered at 0, circular wraps 0 into circular space, as.numeric translates this back into a plain number for analysis

  beta_fatigue    <- rnorm(1, 0, sd_fatigue) #draws trial-level effects from normal distribution (how many values to draw, mean of normal distribution, what standard deviation to use)
  beta_side       <- rnorm(1, 0, sd_side)
  beta_tilt       <- rnorm(1, 0, sd_tilt)
  beta_hemi       <- rnorm(1, 0, sd_hemi)
  beta_sequential <- rnorm(1, 0, sd_sequential)

  acc_i            <- rbeta(1, shape1 = alpha_acc, shape2 = beta_acc) #draw subject's baseline accuracy
  baseline_logodds <- log(acc_i / (1 - acc_i)) #convert accuracy to log-odds so it can be piped into the response model

  # --- SOA: stratified uniform across 61 bins (1.0 to 2.0 s at 1/60 s steps) ---
  # trials split evenly across n_blocks blocks
  trials_per_block <- n_trials / n_blocks #generates how many trials go into each block
  soa_bins  <- seq(1.0, 2.0, by = 1/60)          # generate sequence of 61 possible SOA values between 1 and 2 in steps of 1/60, creates 60 equal soa bins

  block_list <- lapply(seq_len(n_blocks), function(b) { #create one block's data, loop over block number to generate one block's worth of data at a time 
    base_reps <- floor(trials_per_block / length(soa_bins)) #base_reps = floor (75/61) = 1, every soa gets one trial in each block
    remainder <- trials_per_block - base_reps * length(soa_bins) #75-1(61) = 14, 14 soas get a second trial
    soas <- c( #creating a matrix with all the soas each assigned as one base_rep and randomy assigned to remainder trials if needed
      rep(soa_bins, each = base_reps), #each soa_bin gets a base_rep so that each base_rep/soa_bin gets assigned to a trial
      sample(soa_bins, remainder, replace = FALSE) #randomly assigns soa_bin to remainder trials
    )
    soas <- sample(soas)    # randomly select soas in a block
    tibble( #constructs one dataframe per block with columns specified below
      block        = b,
      intended_SOA = soas,
      cue_dir      = constrained_sequence(trials_per_block, max_run = 4),
      tilt_dir     = constrained_sequence(trials_per_block, max_run = 4)
    )
  })

  trial_df      <- bind_rows(block_list) %>% #binds all 6 blocks' dataframes into a singular dataframe
    mutate(trials.thisN = seq_len(n_trials) - 1L) #create a new column from trials.thisN that generates a new numerical sequence counting up all trials, subtract 1 from the count to get back to 0-449 (trial 1 being labeled as 0)

  intended_SOAs <- trial_df$intended_SOA
  # sample timing jitter directly from observed photodiode distribution (no normality assumption)
  if (!is.null(jitter_values)) {
    actual_SOAs <- intended_SOAs + sample(jitter_values, n_trials, replace = TRUE)
  } else {
    actual_SOAs <- intended_SOAs
  }
  trial_index   <- seq_len(n_trials)
  blocks        <- trial_df$block
  cue_dir       <- trial_df$cue_dir
  tilt_dir      <- trial_df$tilt_dir

  # --- generate responses ---
  BRV_accuracy <- numeric(n_trials) #create an array with 450 zeroes and fill them in one by one with BRV_accuracy each run 
  prev_BRV     <- 0   # no previous trial at start

  for (t in seq_len(n_trials)) { #t = trial number, loops through each trial, generating one response at a time

    rhythm_component <- A_i * sin(2 * pi * f_i * actual_SOAs[t] + phi_i) #A_i = amplitude, f_i = how many full cycles (aka frequency, 8 cycles = 8 Hz), phi_i = subject's phase offset (assuming phase NON-alignment)

    trial_effects <-
      beta_fatigue    * (trial_index[t] / n_trials) +
      beta_side       * cue_dir[t] +
      beta_tilt       * tilt_dir[t] +
      beta_hemi       * (cue_dir[t] * tilt_dir[t]) +
      beta_sequential * prev_BRV

    noise_t <- rnorm(1, mean = 0, sd = sigma_noise_i)
    eta     <- baseline_logodds + rhythm_component + trial_effects + noise_t
    p_cor <- logistic(eta)

    response      <- ifelse(runif(1) < p_cor, 1, -1)
    BRV_accuracy[t] <- ifelse(runif(1) < p_nr, NA, response)
    prev_BRV      <- ifelse(is.na(BRV_accuracy[t]), 0, BRV_accuracy[t])
  }

  is_correct <- ifelse(is.na(BRV_accuracy), NA, BRV_accuracy == 1)
  responded  <- !is.na(BRV_accuracy)

  tibble(
    subID        = subID,
    block        = blocks,
    trials.thisN = trial_index - 1L,
    intended_SOA = intended_SOAs,
    actual_SOA   = actual_SOAs,
    BRV_accuracy = BRV_accuracy,
    is_correct   = is_correct,
    responded    = responded,
    cue_dir      = cue_dir,
    tilt_dir     = tilt_dir,
    # store true parameters as metadata columns (dropped before analysis if needed)
    true_A           = A_i,
    true_snr         = snr_i,
    true_sigma_noise = sigma_noise_i,
    true_freq        = f_i,
    true_phase       = phi_i
  )
}

# MAIN SIMULATION WRAPPER

simulate_fsa <- function(
    n_subjects     = 40,
    n_trials       = 450,
    n_blocks       = 6,
    # rhythm
    snr            = 1.0,
    sd_snr         = 0.2,
    sigma_noise    = 0.3,
    sd_noise       = 0.05,
    freq_target    = 8,
    sd_freq        = 0,
    phase_aligned  = FALSE,   # FALSE = uniform random phase; TRUE = all subjects phase = 0
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
    # SOA timing jitter — sampled from observed photodiode distribution (N=540)
    jitter_file    = "~/Desktop/FSA/photodiode_jitter.csv",
    # output
    out_root       = "~/Desktop/FSA/simulated",
    seed           = 42
) {

  set.seed(seed)

  out_root <- path.expand(out_root)

  message(sprintf(
    "Simulating %d subjects x %d trials | SNR=%.2f sd_SNR=%.2f | noise=%.2f sd_noise=%.2f | f=%.1f Hz | phase=%s | p_nr=%.2f",
    n_subjects, n_trials, snr, sd_snr, sigma_noise, sd_noise, freq_target,
    ifelse(phase_aligned, "aligned", "random"), p_nr
  ))

  # load empirical jitter values once and share across all subjects
  jitter_values <- NULL
  if (!is.null(jitter_file) && file.exists(path.expand(jitter_file))) {
    jitter_values <- read.csv(path.expand(jitter_file))$jitter
    message(sprintf("Loaded %d empirical jitter values from %s", length(jitter_values), jitter_file))
  } else {
    message("No jitter file found — actual_SOA = intended_SOA")
  }

  all_data <- map_dfr(seq_len(n_subjects), function(i) {
    subID <- sprintf("SIM%03d", i)
    simulate_subject(
      subID          = subID,
      n_trials       = n_trials,
      n_blocks       = n_blocks,
      snr            = snr,
      sd_snr         = sd_snr,
      sigma_noise    = sigma_noise,
      sd_noise       = sd_noise,
      freq_target    = freq_target,
      sd_freq        = sd_freq,
      phase_aligned  = phase_aligned,
      alpha_acc      = alpha_acc,
      beta_acc       = beta_acc,
      sd_fatigue     = sd_fatigue,
      sd_side        = sd_side,
      sd_tilt        = sd_tilt,
      sd_hemi        = sd_hemi,
      sd_sequential  = sd_sequential,
      p_nr           = p_nr,
      jitter_values  = jitter_values
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
      true_A           = first(true_A),
      true_snr         = first(true_snr),
      true_sigma_noise = first(true_sigma_noise),
      true_freq        = first(true_freq),
      true_phase       = first(true_phase),
      n_trials         = n(),
      pct_responded    = mean(responded) * 100,
      accuracy         = mean(is_correct, na.rm = TRUE) * 100,
      .groups          = "drop"
    )

  cat("\n=== SIMULATION SUMMARY ===\n")
  cat(sprintf("Output saved to: %s\n", out_root))
  cat(sprintf("Mean SNR:        %.3f (SD %.3f)\n",
              mean(summary_df$true_snr), sd(summary_df$true_snr)))
  cat(sprintf("Mean noise (SD): %.3f (SD %.3f)\n",
              mean(summary_df$true_sigma_noise), sd(summary_df$true_sigma_noise)))
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
# simulate_fsa(snr = 0, sd_snr = 0, out_root = "~/Desktop/FSA/simulated/null")

# Single frequency, aligned phases, moderate SNR
# simulate_fsa(snr = 1.0, sd_snr = 0.2, sigma_noise = 0.3, freq_target = 8, phase_aligned = TRUE,
#              out_root = "~/Desktop/FSA/simulated/signal_aligned")

# Single frequency, random phases (tests phase-agnostic detection)
# simulate_fsa(snr = 1.0, sd_snr = 0.2, sigma_noise = 0.3, freq_target = 8, phase_aligned = FALSE,
#              out_root = "~/Desktop/FSA/simulated/signal_random_phase")

# Low SNR (weak signal)
# simulate_fsa(snr = 0.3, sd_snr = 0.1, sigma_noise = 0.3,
#              out_root = "~/Desktop/FSA/simulated/signal_low_snr")

# High SNR (strong signal)
# simulate_fsa(snr = 2.0, sd_snr = 0.2, sigma_noise = 0.3,
#              out_root = "~/Desktop/FSA/simulated/signal_high_snr")

# With between-subject frequency variability
# simulate_fsa(snr = 1.0, sd_snr = 0.2, sigma_noise = 0.3, freq_target = 8, sd_freq = 2,
#              out_root = "~/Desktop/FSA/simulated/signal_freq_variability")

# With trial-level confounds
# simulate_fsa(snr = 1.0, sd_snr = 0.2, sigma_noise = 0.3, freq_target = 8,
#              sd_fatigue = 0.2, sd_sequential = 0.15,
#              out_root = "~/Desktop/FSA/simulated/signal_with_confounds")

# simulate_fsa()

