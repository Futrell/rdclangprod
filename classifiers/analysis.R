setwd("~/projects/control/code/classifiers")
rm(list=ls())
library(tidyverse)
library(lme4)
library(plotrix)
library(showtext)
library(latex2exp)
showtext_auto()


CORPUS_DATA = "heldout_data_complete_20171128.csv" # why do some nouns have multiple distinct log_noun_freqs?
QUICK_EXPERIMENTAL_DATA = "data_with_transcript_and_noun_surprisal_expt_20190114.csv"
SLOW_EXPERIMENTAL_DATA = "data_with_transcript_and_noun_surprisal_expt_20190127.csv"

GENERIC = "个"

raw_d = read_csv(CORPUS_DATA) 

d = raw_d %>%
  mutate(is_spec=outcome == "specific") %>%
  group_by(specl) %>%
    mutate(n_specl=n()) %>%
    ungroup() %>%
  group_by(noun, specl) %>%
    mutate(log_noun_freq=log(sum(exp(log_noun_freq)))) %>%
    ungroup() # >:(

write_csv(d, "heldout_data_merged.csv")
  
specl_stats = d %>% 
  select(specl, n_specl) %>%
  distinct() %>%
  arrange(-n_specl) %>%
  mutate(specl_rank=1:nrow(.)) %>%
  mutate(specl=factor(specl, levels=pull(., specl))) 

d = d %>% 
  inner_join(specl_stats) %>%
  mutate(specl=factor(specl, levels=levels(specl_stats$specl)))

specl_stats %>% 
  filter(specl_rank<40) %>%
  ggplot(aes(x=specl_rank, y=log(n_specl), label=specl)) + 
    geom_text() +
    theme_classic() +
    labs(x="Frequency Rank",
         y="Log Frequency")

d %>% 
  select(specl, n_specl, specl_rank, noun) %>%
  distinct() %>%
  group_by(specl) %>%
    summarize(num_nouns=n()) %>%
    ungroup() %>%
  inner_join(specl_stats) %>%
  ggplot(aes(x=log(n_specl), y=num_nouns, label=specl)) +
    geom_text() +
    theme_classic()

specificity = d %>%
  filter(specl_rank<30) %>%
  group_by(specl, noun) %>%
    summarize(n=n()) %>%
    ungroup() %>%
  group_by(specl) %>%
    mutate(Z=sum(n)) %>%
    ungroup() %>%
  mutate(p_noun_given_specl=n/Z) %>%
  group_by(specl) %>%
    summarize(entropy=-sum(p_noun_given_specl * log(p_noun_given_specl))) %>%
    ungroup() %>%
  arrange(entropy) %>%
  mutate(entropy_rank=1:nrow(.)) 

specificity %>%
  ggplot(aes(x=entropy_rank, y=entropy, label=specl)) +
    geom_text() +
    theme_classic()

specificity %>%
  inner_join(specl_stats) %>%
  ggplot(aes(x=log(n_specl), y=entropy, label=specl)) +
    geom_text() + 
    theme_classic()
  
d %>%
  group_by(noun) %>%
    summarize(n=n()) %>%
    ungroup() %>%
  arrange(-n) %>%
  mutate(rank=1:nrow(.)) %>%
  ggplot(aes(x=rank, y=log(n), label=noun)) +
    geom_text() +
    theme_classic()
  

plot_surprisal = function(d, num_bins) {
  d %>%
    group_by(specl) %>%
      mutate(surprisal_bin=cut_interval(noun_surprisal, num_bins)) %>%
      ungroup() %>%
    group_by(surprisal_bin, specl) %>%
      summarize(m_outcome=mean(is_spec),
                noun_surprisal=mean(noun_surprisal),
                se=std.error(is_spec),
                upper=min(1, m_outcome+1.96*se),
                lower=max(0, m_outcome-1.96*se)) %>%
      ungroup() %>%
    ggplot(aes(x=noun_surprisal, y=m_outcome, color=specl, label=specl)) +
      geom_errorbar(aes(ymin=lower, ymax=upper)) +
      geom_line() +
      ylim(0,1) +
      theme_classic() +
      theme(legend.position="none") +
      xlab("Noun surprisal") +
      ylab("P(specific classifier)")
}

plot_freq = function(d, num_bins) {
  d %>%
    group_by(specl) %>%
      mutate(surprisal_bin=cut_interval(log_noun_freq, num_bins)) %>%
      ungroup() %>%
    group_by(surprisal_bin, specl) %>%
      summarize(m_outcome=mean(is_spec),
                noun_surprisal=-mean(log_noun_freq),
                se=std.error(is_spec),
                upper=min(1, m_outcome+1.96*se),
                lower=max(0, m_outcome-1.96*se)) %>%
      ungroup() %>%
    ggplot(aes(x=noun_surprisal, y=m_outcome, label=specl, color=specl)) +
      geom_errorbar(aes(ymin=lower, ymax=upper)) +
      geom_line() +
      ylim(0,1) +
      theme_classic() +
      theme(legend.position="none") +
      xlab("Noun unigram surprisal") +
      ylab("P(specific classifier)")
}
 
d %>% 
  filter(noun_surprisal < 20,
         n_specl > 2000) %>%
  plot_freq(num_bins=5) + geom_text(color="black")

d %>% 
  filter(noun_surprisal < 22,
         specl_rank < 17) %>%
  plot_freq(num_bins=10) + facet_wrap(~specl)

plot_model_predictions = function(d, num_bins=7, point_alpha=.1, line_alpha=.1, line_size=1, legend_position=c(.85,.2)) {
d %>%
  group_by(gain) %>%
    mutate(bin=cut_interval(lnp_g, num_bins)) %>%
    ungroup() %>%
  group_by(gain, bin) %>%
    summarize(m=mean(p_spec),
              se=std.error(p_spec),
              upper=min(m+1.96*se, 1),
              lower=max(m-1.96*se, 0),
              lnp_g=mean(lnp_g)) %>%
    ungroup() %>%
  ggplot(aes(x=lnp_g / log(2), y=m, color=gain)) + 
    geom_point(data=dm, aes(y=p_spec), alpha=point_alpha) +
    geom_line(data=dm, aes(y=p_spec, group=interaction(gain, classifier)), alpha=line_alpha) +
    geom_line(size=line_size) +
    geom_errorbar(aes(ymin=lower, ymax=upper), width=0) +
    theme_classic() +
    theme(legend.position=legend_position) +
    labs(x=TeX("$\\Log_2$ Need Probability"),
         y="P(specific classifier)",
         color=TeX("Control gain $\\alpha$"))
}

plot_empirical = function(d, num_bins=5, point_alpha=.2, line_alpha=.1, line_size=1, legend_position=c(.9, .3)) {
d %>%
  group_by(condition) %>%
  mutate(bin=cut_interval(log_freq, num_bins)) %>%
  ungroup() %>%
  group_by(condition, bin) %>%
  summarize(se=std.error(p_spec),
            p_spec=mean(p_spec),
            log_freq=mean(log_freq),
            upper=min(1, p_spec+1.96*se),
            lower=max(0, p_spec-1.96*se)) %>%
  ungroup() %>%
  ggplot(aes(x=log_freq/log(2), y=p_spec, color=condition)) +
  geom_text(aes(label=preferred_spec_cl), data=d, alpha=point_alpha) +
  geom_line(aes(group=interaction(condition, preferred_spec_cl)), data=d, alpha=line_alpha) +
  geom_line(size=line_size) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=0) +
  theme_classic() +
  labs(x=TeX("$\\Log_2$ Noun Frequency"),
       y="P(specific classifier)",
       color="Condition") +
  theme(legend.position=legend_position)
}

plot_model_entropy = function(d, which_gain="1.1", legend_position=c(.25, .35), line_alpha=.5) {
  d %>%
    filter(gain==which_gain) %>%
    ggplot(aes(x=H_spec / log(2), y=p_spec, color=lnp_g / log(2), group=classifier)) +
    #stat_smooth() +
    geom_point() +
    geom_line(alpha=line_alpha) +
    theme_classic() +
    labs(x="Entropy over specific classifier (bits)",
         y="P(specific classifier)",
         color=TeX("$\\Log_2$ Need Probability")) +
    theme(legend.position=legend_position,
          legend.background = element_rect(fill='transparent')) 
}



de = read_csv(QUICK_EXPERIMENTAL_DATA) %>% 
  mutate(condition="quick") %>%
  bind_rows(read_csv(SLOW_EXPERIMENTAL_DATA) %>% mutate(condition="slow")) %>%
  mutate(worker_id = as.factor(worker_id)) %>%
  filter(noun_used == most_freq_used_noun,
         classifier_used %in% c(preferred_spec_cl, GENERIC)) %>%
  mutate(is_spec=classifier_used == preferred_spec_cl) %>%
  mutate(log_freq=log(most_produced_noun_frequency_sogouw))

de_prop = de %>%
  group_by(stimulus, preferred_spec_cl, log_freq, condition) %>%
  summarize(p_spec=mean(is_spec),
            se=std.error(p_spec),
            upper=min(1, p_spec+1.96*se),
            lower=max(0, p_spec-1.96*se)) %>%
  ungroup()


de_prop %>% 
plot_empirical(num_bins=5, line_alpha=.1, point_alpha=.3, legend_position=c(.85,.2)) +
  ggtitle("A. Zhan & Levy (2019) Experiment")
ggsave("zl2019_empirical.pdf", width=4, height=4)


dm = read_csv("classifier_model.csv") %>%
  mutate(gain=as.factor(gain), classifier=as.factor(classifier))
plot_model_predictions(dm, num_bins=5, line_alpha=.1, point_alpha=.2, legend_position=c(.85,.2)) +
  ylim(0,1) +
  ggtitle("B. Model Simulation")
ggsave("zl2019_model.pdf", width=4, height=4)

plot_model_entropy(dm, legend_position=c(.27, .35), which_gain="1.1") +
  ggtitle("C. Model Entropy Effect") + ylim(0,1) + xlim(0, NA)
ggsave("zl2019_entropy.pdf", width=4, height=4)

dm %>%
  ggplot(aes(x=H_spec, y=p_spec, color=gain, group=interaction(gain, classifier))) +
    geom_point() +
    geom_line(alpha=.5) +
    theme_classic() +
    labs(x="Entropy over specific classifier",
         y="P(specific classifier)",
         color=TeX("Control gain $\\alpha$")) +
    theme(legend.position=c(.4, .2))



m = glmer(is_spec ~ 
            I(log(most_produced_noun_frequency_sogouw)) + condition
          + (1+I(log(most_produced_noun_frequency_sogouw)) + condition | worker_id) +
            (1+I(log(most_produced_noun_frequency_sogouw)) + condition | preferred_spec_cl),
          data=de, family="binomial")

N = 200
K = 10
z = data.frame(cl=rep(1:K, each=N/K)) %>%
  sample_frac(1) %>%
  mutate(cl=as.factor(cl), n=1:N) %>%
  mutate(a=1/n, Z=sum(a), p=a/Z)

z %>%
  ggplot(aes(x=n, y=p, fill=cl)) +
    geom_bar(stat="identity") +
    theme_classic() +
    theme(legend.position="none", axis.ticks.x=element_blank(), axis.text.x=element_blank()) +
    labs(x="Noun", y="P(Noun)") +
    xlim(0, 50)
  
