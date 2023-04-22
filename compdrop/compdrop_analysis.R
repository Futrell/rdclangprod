rm(list=ls())
setwd("~/projects/control/code/compdrop")

library(tidyverse)
library(viridis)
library(latex2exp)
library(brms)
library(cowplot)

grid = read_csv("compdrop_sims.csv")

grid %>% 
  ggplot(aes(x=gain, y=discount, fill=freq_diff, color=freq_diff)) + 
    geom_tile() +
    scale_fill_gradient2() + 
    scale_color_gradient2() +
    theme_classic() +
    xlim(0, 3) +
    labs(x=TeX("Gain $\\alpha$"),
        y=TeX("Discount $\\gamma$"),
        fill="", color="") +
    ggtitle("A. High-surprisal complementizer preference")

ggsave("compdrop_surp.pdf", width=5.5, height=3)

grid %>% 
  ggplot(aes(x=gain, y=discount, fill=freq_diff2, color=freq_diff2)) + 
  geom_tile() +
  scale_fill_gradient2() + 
  scale_color_gradient2() +
  theme_classic() +
  xlim(0, 3) +
  labs(x=TeX("Gain $\\alpha$"),
       y=TeX("Discount $\\gamma$"),
       fill="", color="") +
  ggtitle("B. High-surprisal complementizer preference (contextual)")


ggsave("compdrop_surp2.pdf", width=5.5, height=3)

data = read_csv("lm_output.csv")

d = data %>%
  filter(complementizer %in% c("that", "which", "who", "whom", "THAT", "That", "Which", "-NONE-")) %>%
  mutate(that = complementizer != "-NONE-")

d %>%
  ggplot(aes(x=that, y=entropy)) + 
  geom_boxplot() +
  geom_jitter(alpha=.2) +
  theme_classic() +
  labs(x="Explicit complementizer?",
       y="Entropy (nats)")

d %>%
  ggplot(aes(x=that, y=-lnp_both)) + 
    geom_boxplot() +
    geom_jitter(alpha=.2) +
    theme_classic() +
    labs(x="Explicit complementizer?",
        y="Surprisal (nats)")

m = d %>%
  brm(that ~ lnp_both + (1+lnp_both | NPhead), family="bernoulli", data=.)



             

