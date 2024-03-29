rm(list=ls())
setwd("~/projects/control/code/goodenough")
library(tidyverse)
library(broom)
library(lme4)
library(brms)
library(plotrix)
library(pracma)
library(grid)
library(gridExtra)
library(cowplot)
library(latex2exp)
library(bayestestR)


# Reward is determined by a von Mises distribution,
# resulting in reward differential d(w, x) - d(w, y) where d is cosine similarity.

EPSILON = .001

GGRED = "#F8766D"
GGGREEN = "#00BA38"
GGBLUE = "#619CFF"
GGPURPLE = "#C77CFF"
GGYELLOW = "#CD6900"

cosine_diff = function(theta1, theta2) {
  theta1r = theta1 * (pi/180)
  theta2r = theta2 * (pi/180)
  cos(theta1r - theta2r)
}

preprocess = function(dd) {
dd %>%
  filter(trialType == "test",
         block != "test_x",
         valid_label == 1) %>%
  mutate(distance1=cosine_diff(angle, nearbyAngle1),
         distance2=cosine_diff(angle, nearbyAngle2),
         rel_distance=distance1 - distance2,
         abs_rel_distance=abs(rel_distance),
         is1=label == nearbyLabel1,
         is2=label == nearbyLabel2,
         distance_target=if_else(rel_distance>0, distance1, distance2),
         distance_distractor=if_else(rel_distance>0, distance2, distance1),
         target_freq=if_else(rel_distance>0, nearbyFreq1, nearbyFreq2),
         is_correct=if_else(rel_distance>0, label == nearbyLabel1, label == nearbyLabel2),
         freqdiff1=if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "lf", +1,
                           if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "hf", -1, 0)),
         freqdiff=if_else(rel_distance>0,
                          if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "lf", +1,
                            if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "hf", -1, 0)),
                          if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "lf", -1,
                            if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "hf", +1, 0))),
         freqprof=if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "hf", "hf_hf",
                  if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "lf", "lf_lf",
                  if_else(rel_distance>0, 
                          if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "lf", "hf_lf",
                            if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "hf", "lf_hf", "!")),
                          if_else(nearbyFreq1 == "hf" & nearbyFreq2 == "lf", "lf_hf",
                            if_else(nearbyFreq1 == "lf" & nearbyFreq2 == "hf", "hf_lf", "!")))))) %>%
  mutate(freqprof=factor(freqprof, levels=c("lf_lf", "lf_hf", "hf_lf", "hf_hf"))) %>%
    filter(!is.na(is_correct) & !is.na(rel_distance) & !is.na(freqdiff))
}  

d2 = preprocess(read_csv("Words_v2_final_preprocessed.csv")) %>%
  mutate(room=as.character(room))
d3 = preprocess(read_csv("Words_v3_final_preprocessed.csv"))
d4 = preprocess(read_csv("Words_v4_final_preprocessed.csv"))
d_all = preprocess(read_csv("Words_final_preprocessed.csv")) %>%
  filter(expVersion %in% c("wc2_2_v1", "wc2_4_v1", "wc2_3_v1"))
d = d4


## ABSOLUTE DISTANCE ANALYSES
         
dp_distance1 = d %>%
  group_by(distance1, nearbyFreq1) %>%
    summarize(p=mean(label == nearbyLabel1)) %>%
    ungroup()

# Theory predicts a linear relationship with same slope
dp_distance1 %>%
  ggplot(aes(x=distance1, y=log(p), color=nearbyFreq1)) +
    geom_point() +
    stat_smooth() +
    theme_classic() +
    xlab("R_g(x)") +
    ylab("ln p_g(x)")


# TWO-ALTERNATIVE ANALYSES

# for manual logistic regression using optimize:
log_loss = function(d, alpha, beta) {
  yhat = with(d, sigmoid(alpha*freqdiff1 + beta*rel_distance))
  ll = with(d, is1*log(yhat) + (1-is1)*log(1-yhat))
  -sum(ll)
}

HF_LF = "x HF, y LF"
LF_HF = "x LF, y HF"
SAME = "Same frequency"
ORDER = c(LF_HF, SAME, HF_LF)

fit_plot = function(fit_data, eval_data) {

# fit models to d4; theory does not accommodate an intercept
m_rel = glm(is1 ~ 0 + rel_distance + freqdiff1, data=fit_data, family="binomial")

intercept_rel = 0 #tidy(m_rel) %>% filter(term == "(Intercept)") %>% pull(estimate)
gamma_rel = tidy(m_rel) %>% filter(term == "rel_distance") %>% pull(estimate)
offset_rel = tidy(m_rel) %>% filter(term == "freqdiff1") %>% pull(estimate)

print(gamma_rel)
print(offset_rel)

boundaries = data.frame(lf=offset_rel/gamma_rel,
                        hf=-offset_rel/gamma_rel, 
                        same=0) %>%
  gather(freqdiff1, decision_boundary) %>%
  mutate(freqdiff1=if_else(freqdiff1=="lf", LF_HF, if_else(freqdiff1=="hf", HF_LF, SAME)),
         freqdiff_factor=factor(freqdiff1, levels=ORDER)) %>%
  select(-freqdiff1) 

rdc_rel = linspace(min(fit_data$rel_distance), max(fit_data$rel_distance), 100) %>%
  expand.grid(c("LF_LF", "LF_HF", "HF_LF", "HF_HF")) %>%
  rename(rel_distance=Var1, conditionType=Var2) %>%
  mutate(freqdiff1=if_else(conditionType == "LF_HF", +1, if_else(conditionType == "HF_LF", -1, 0))) %>% # beware signs
  mutate(freqdiff_factor=if_else(freqdiff1 == +1, HF_LF, if_else(freqdiff1 == -1, LF_HF, SAME)),
         freqdiff_factor=factor(freqdiff_factor, levels=ORDER)) %>%
  mutate(p=sigmoid(intercept_rel + gamma_rel*rel_distance + offset_rel*freqdiff1),
         type="Model")

eval_data %>%
  mutate(R_bin=cut_width(rel_distance, max(rel_distance)/4)) %>%
  group_by(R_bin) %>%
    mutate(rel_distance=mean(rel_distance)) %>%
    ungroup() %>%
  group_by(rel_distance, freqdiff1) %>%
    summarize(p=mean(is1),
            se=std.error(is1),
            upper=min(p+1.96*se, 1),
            lower=max(p-1.96*se, 0)) %>%
    ungroup() %>%
  mutate(type="Experiment") %>%
  mutate(freqdiff_factor=if_else(freqdiff1==+1, HF_LF, if_else(freqdiff1==-1, LF_HF, SAME)),
         freqdiff_factor=factor(freqdiff_factor, levels=ORDER)) %>%
  inner_join(boundaries) %>%
  ggplot(aes(x=rel_distance, 
             y=p,
             color=freqdiff_factor,
             group=freqdiff_factor
  )) +
  geom_vline(linetype="dashed", aes(xintercept=decision_boundary, color=freqdiff_factor)) +
  scale_y_continuous(sec.axis = sec_axis(~., name="Proportion x responses")) +
  geom_line(data=rdc_rel, aes(xintercept=0)) +
  geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=0) +
  scale_color_manual(values=c(GGRED, GGGREEN, GGBLUE), labels=c(TeX("$x$ LF, $y$ HF"), 
                                                                "Same freq.", 
                                                                TeX("$x$ HF, $y$ LF"))) +
  theme_classic() +
  theme(legend.position=c(.22, .7), 
        legend.title=element_blank(),
        legend.background = element_rect(fill='transparent')) +
  annotate("text", x=-.1, y=1.1, label=TeX("favors $y \\leftarrow$")) +
  annotate("text", x=+.1, y=1.1, label=TeX("$\\rightarrow$ favors $x$")) +
  xlab(TeX("Reward Differential $\\Delta R_g$")) +
  ylab(TeX("Policy Probability $\\pi_g(x)$"))
}

fit_plot(d4, d3) + ggtitle("B. Probability to produce word x")
ggsave("../plots/differential.pdf", width=3.5, height=3.25)


fit_plot(d3, d4) + ggtitle("Fit to Exp 3, Test on Exp 4")
ggsave("../plots/differential_34.pdf", width=3.5, height=3.25)
fit_plot(d4, d3) + ggtitle("Fit to Exp 4, Test on Exp 3")
ggsave("../plots/differential_43.pdf", width=3.5, height=3.25)







# Model for statistical tests; apply to all data for maximum power
m_rel_int0 = glmer(is1 ~ rel_distance + freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d_all, family="binomial")
m_rel_int = glmer(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d_all, family="binomial")
anova(m_rel_int0, m_rel_int) # no interaction

m_rel4_int0 = glmer(is1 ~ rel_distance + freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d4, family="binomial")
m_rel4_int = glmer(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d4, family="binomial")
anova(m_rel4_int0, m_rel4_int) # no interaction

m_rel3_int0 = glmer(is1 ~ rel_distance + freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d3, family="binomial")
m_rel3_int = glmer(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d3, family="binomial")
anova(m_rel3_int0, m_rel3_int) # no interaction

m_rel2_int0 = glmer(is1 ~ rel_distance + freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d2, family="binomial")
m_rel2_int = glmer(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1||subjCode), data=d2, family="binomial")
anova(m_rel2_int0, m_rel2_int) # no interaction



# R2: The evidence against an interaction here (in line 230) is quite weak: 
# given the breadth of the HPD interval, there is quite a bit of probability 
# mass outside of any plausible region of practical equivalence (ROPE) here. 
# I think this is an important caveat. It is also I think important to show 
# how much of the probability mass in the HPD is in the positive region. 

m_rel_intb = brm(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1|subjCode), data=d_all, family="bernoulli")

#m_rel2_intb = brm(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1|subjCode), data=d2, family="bernoulli")
#m_rel3_intb = brm(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1|subjCode), data=d3, family="bernoulli")
#m_rel4_intb = brm(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1|subjCode), data=d4, family="bernoulli")

plot(m_rel_intb, variable = "b_rel_distance:freqdiff1")
equivalence_test(m_rel_intb, ci=.89)

#m_rel_intb = brm(is1 ~ rel_distance * freqdiff1 + (1+rel_distance*freqdiff1|subjCode), data=d_all, family="bernoulli")


# TWO-ALTERNATIVE RELATIVE DISTANCE ACCURACY ANALYSES

# Model for parameter fitting
m = glm(is_correct ~ 0 + abs_rel_distance + freqdiff, data=d4, family="binomial")

intercept = 0
gamma = tidy(m) %>% filter(term == "abs_rel_distance") %>% pull(estimate)
offset = tidy(m) %>% filter(term == "freqdiff") %>% pull(estimate)


rdc = linspace(0, max(d$abs_rel_distance), 100) %>%
  expand.grid(c("lf_lf", "lf_hf", "hf_lf", "hf_hf")) %>%
  rename(abs_rel_distance=Var1, freqprof=Var2) %>%
  mutate(freqdiff=if_else(freqprof == "lf_hf", -1,
                  if_else(freqprof == "hf_lf", +1, 0))) %>%
  mutate(p=sigmoid(intercept + gamma*abs_rel_distance + offset*freqdiff),
         type="Model") %>%
  mutate(freqprof=if_else(freqprof=="lf_lf", "LF target\n LF distractor",
                  if_else(freqprof=="hf_hf", "HF target\n HF distractor",
                  if_else(freqprof=="hf_lf", "HF target\n LF distractor",
                  if_else(freqprof=="lf_hf", "LF target\n HF distractor", "!")))))


d3 %>%
  mutate(R_bin=cut_width(abs_rel_distance, max(abs_rel_distance)/7)) %>%
  group_by(R_bin) %>%
    mutate(abs_rel_distance=median(abs_rel_distance)) %>%
    ungroup() %>%
  group_by(abs_rel_distance, freqdiff, freqprof) %>%
  summarize(p=mean(is_correct),
            se=std.error(is_correct),
            upper=min(p+1.96*se, 1),
            lower=p-1.96*se) %>%
  ungroup() %>%
  mutate(type="Experiment") %>%
  mutate(freqprof=if_else(freqprof=="lf_lf", "LF target\n LF distractor",
                  if_else(freqprof=="hf_hf", "HF target\n HF distractor",
                  if_else(freqprof=="hf_lf", "HF target\n LF distractor",
                  if_else(freqprof=="lf_hf", "LF target\n HF distractor", "!"))))) %>%
  ggplot(aes(x=abs_rel_distance, 
             y=p,
             color=freqprof
  )) +
  geom_line(data=rdc) +
  #geom_line(linetype="dashed", color="gray") +
  geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=0) +
  scale_color_manual(values=c(GGPURPLE, GGBLUE, GGRED, GGYELLOW)) +
  theme_classic() +
  theme(legend.position="none") +
  facet_grid(~freqprof) +
  xlab(TeX("$| \\Delta R_g |$")) +
  ylab("") +
  ggtitle("C. Probability to produce more accurate word")

ggsave("../plots/accuracy.pdf", width=4.2, height=3.25)
