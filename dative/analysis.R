setwd("~/projects/control/code/dative")
rm(list=ls())
library(lme4)
library(brms)
library(languageR)
library(tidyverse)

datives = read_csv("datives-finalized.csv") 
dg = read_csv("datives_flipped_gpt3.csv") %>% rename(Token.ID=id)

d = inner_join(datives, dg) %>%
  mutate(DO=Response.variable == "D") %>%
  mutate(Semantics.reduced=if_else(Semantics == "C", "A", Semantics)) %>%
  mutate(Recipient.definiteness.numeric=case_when(
    Recipient.definiteness == "Definite-pn" ~ 2,
    Recipient.definiteness == "Definite" ~ 1,
    Recipient.definiteness == "Indefinite" ~ 0
  )) %>%
  mutate(Theme.definiteness.numeric=case_when(
    Theme.definiteness == "Definite-pn" ~ 2,
    Theme.definiteness == "Definite" ~ 1,
    Theme.definiteness == "Indefinite" ~ 0
  )) %>%
  mutate(Recipient.animacy.numeric=case_when(
    Recipient.animacy == "A" ~ 1,
    Recipient.animacy == "C" ~ 0,
    Recipient.animacy == "L" ~ 0,
    Recipient.animacy == "I" ~ 0    
  )) %>% 
  mutate(Theme.animacy.numeric=case_when(
    Theme.animacy == "A" ~ 1,
    Theme.animacy == "C" ~ 0,
    Theme.animacy == "L" ~ 0,
    Theme.animacy == "I" ~ 0    
  )) %>%
  mutate(Theme.charlength=str_length(Theme),
         Recipient.charlength=str_length(Recipient)) %>%
  mutate(
    D.definiteness = Recipient.definiteness.numeric - Theme.definiteness.numeric,
    D.animacy = Recipient.animacy.numeric - Theme.animacy.numeric,
    D.logprob_both = logprob_D_both - logprob_P_both,
    D.logprob_first = logprob_recipient_in_context - logprob_theme_in_context,
    D.logprob_plan_both = D.logprob_both - D.logprob_first,
    D.length = Recipient.length - Theme.length,
    D.charlength = Recipient.charlength - Theme.charlength
  )


ml = glmer(DO ~ D.definiteness + 
             D.animacy + 
             D.charlength +
             D.logprob_first + 
             D.logprob_plan_both + 
             Semantics.reduced + 
             (1 | Speaker), 
        data=d, family="binomial")

m = brm(DO ~ 
          D.definiteness + 
          D.animacy + 
          D.charlength +
          D.logprob_first + 
          D.logprob_plan_both + 
          Semantics.reduced + 
          (1 + 
             D.definiteness + 
             D.animacy + 
             D.charlength +
             D.logprob_first + 
             D.logprob_plan_both + 
             Semantics.reduced | Speaker), 
        data=d, family="bernoulli")

hypothesis(m, "D.logprob_first = 0")
hypothesis(m, "D.logprob_plan_both = 0")
hypothesis(m, "D.logprob_first - D.logprob_plan_both > 0")



