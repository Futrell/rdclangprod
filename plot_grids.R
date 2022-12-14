library(tidyverse)
library(viridis)
library(latex2exp)

setwd("~/projects/control/code")

# SHORT-BEFORE-LONG PREFERENCE

dsl = read_csv("output/shortlong.csv")

dsl %>% 
  filter(gamma>=1) %>%
  #mutate(statistic=log(p_short) - log(p_long)) %>%
  mutate(statistic=p_short/(p_short + p_long)) %>%
  mutate(statistic=if_else(is.infinite(statistic), NaN, statistic)) %>%
  mutate(statistic=if_else(is.na(statistic), max(statistic, na.rm=T), statistic)) %>%
  ggplot(aes(x=gamma, y=alpha, color=statistic)) + 
  geom_point(shape=15) +
  theme_classic() +
  #scale_color_gradient2(low="red", high="blue", midpoint=1/2) +
  scale_color_viridis() +
  ylim(0, 1) +
  labs(x=TeX("Gain $\\gamma$"), 
       y=TeX("Discount $\\alpha$"),
       #color=TeX("$\\ln \\frac{p_{g_1}(a)}{p_{g_1}(b)}$")) +
       color=TeX("$\\frac{p_{g_1}(short)}{p_{g_1}(short) + p_{g_1}(long)}$")) +
  theme(legend.title=element_text(size=7)) +
  ggtitle("Short-before-long preference")

ggsave("plots/shortlong.pdf", width=5, height=3.3)


# FILLED PAUSE

dfl_policy = read_csv("output/fp_policy.csv") %>%
  select(-`...1`) %>%
  mutate(p_L=exp(R_g - log(5)))

dfl_policy %>%
  ggplot(aes(x=factor(x, levels=c("5", "4", "3", "2", "1", "0")),
             y=as.factor(g), 
             fill=p_L,
             color=p_L)) +
  geom_tile() +
  scale_x_discrete(labels=c(
    TeX("$x_5$"),
    TeX("$x_4$"), 
    TeX("$x_3$"), 
    TeX("$x_2$"),
    TeX("$x_1$"), 
    TeX("$e$")
  )) +
  scale_fill_viridis() +
  scale_color_viridis() +
  coord_flip() +
  labs(x="Action x", y="Goal g", fill=TeX("$p_L(w_g | x)$")) +
  guides(color="none") +
  theme_classic() +
  ggtitle(TeX("A. Listener model"))

#ggsave("plots/fp_listener.pdf", height=4, width=3.5)

dfl_policy %>%
  ggplot(aes(x=factor(x, levels=c("5", "4", "3", "2", "1", "0")),
             y=as.factor(g), 
             fill=R_g)) +
  geom_tile(color="black") +
  scale_x_discrete(labels=c(
    TeX("$x_5$"),
    TeX("$x_4$"), 
    TeX("$x_3$"), 
    TeX("$x_2$"),
    TeX("$x_1$"), 
    TeX("$e$")
  )) +
  scale_fill_gradient2(low="blue", high="red", midpoint=0) +
  coord_flip() +
  labs(x="Action x", y="Goal g", fill=TeX("$R_g(x)$")) +
  guides(color="none") +
  theme_empty() +
  ggtitle(TeX("A. Reward with filled pause e"))

ggsave("plots/fp_reward.pdf", height=4, width=3.5)


dfl_policy %>%
  ggplot(aes(x=factor(x, levels=c("5", "4", "3", "2", "1", "0")),
             y=as.factor(g), 
             fill=exp(`lnp_g(x)`),
             color=exp(`lnp_g(x)`))) +
    geom_tile() +
  scale_x_discrete(labels=c(
    TeX("$x_5$"),
    TeX("$x_4$"), 
    TeX("$x_3$"), 
    TeX("$x_2$"),
    TeX("$x_1$"), 
    TeX("$e$")
  )) +
  scale_fill_viridis() +
  scale_color_viridis() +
  coord_flip() +
  labs(x="Action x", y="Goal g", fill=TeX("$\\ln p_g(x)$")) +
  guides(color="none") +
  theme_classic() +
  ggtitle(TeX("B. Policy with filled pause $e$"))

#ggsave("plots/fp_policy.pdf", height=4, width=3.5)

dfl_policy %>%
  ggplot(aes(x=factor(x, levels=c("5", "4", "3", "2", "1", "0")),
             y=as.factor(g), 
             fill=exp(`lnp_g(x)`),
             color=exp(`lnp_g(x)`))) +
  geom_tile() +
  scale_x_discrete(labels=c(
    TeX("$x_5$"),
    TeX("$x_4$"), 
    TeX("$x_3$"), 
    TeX("$x_2$"),
    TeX("$x_1$"), 
    TeX("$e$")
  )) +
  scale_fill_viridis() +
  scale_color_viridis() +
  coord_flip() +
  labs(x="Action x", y="Goal g", fill=TeX("$\\ln p_g(x)$")) +
  guides(color="none") +
  theme_classic() +
  ggtitle(TeX("B. Policy with filled pause $e$"))


dfl = read_csv("output/fp_summary.csv") %>%
  select(-`...1`)

dfl %>%
  gather(measure, value, -DR) %>%
  filter(measure != "p0(x)") %>%
  ggplot(aes(x=DR, y=log(value), color=measure)) +
    geom_line(size=1) +
    geom_point() +
    theme_classic() +
    labs(x=TeX("$\\Delta R_{g}$"),
         y="Log Probability") +
    guides(color="none") +
    annotate("text", x=1.75, y=-.7, color="black", hjust=0, label=TeX("$p_{g}(x_g)$"), group=1) +
    annotate("text", x=1.75, y=-1.6, color="black", hjust=0, label=TeX("$p_0(x_g | e)$"), group=1) +
    annotate("text", x=1.75, y=-3.1, color="black", hjust=0, label=TeX("$p_{g}(e)$"), group=1) +
    ggtitle("B. Effect of signal strength")

#ggsave("plots/fp_strength.pdf", height=4, width=3)

dfl %>%
  select(-`p0(x)`, -`p0(x|??)`) %>%
  ggplot(aes(x=DR, y=`p_{g_x}(??)`, color=`p_{g_x}(x)`)) +
    geom_line(size=2) +
    geom_point(size=3) +
    theme_classic() +
    labs(x=TeX("$\\Delta R_{g}$"),
         y=TeX("Controlled $p_g(e)$"),
         color=TeX("$p_g(x_g)$")) +
    theme(legend.position=c(.28, .3)) +
    scale_color_viridis() +
    ggtitle("B. Filled pause probability")

ggsave("plots/fp_prob.pdf", height=4, width=3)



dfl %>%
  ggplot(aes(x=`p_{g_x}(x)`, y=`p0(x|??)`, color=DR)) +
  geom_point(size=3) +
  geom_line(size=2) +
  theme_classic() +
  scale_color_viridis() +
  labs(color=TeX("$\\Delta R_{g}$"),
       y=TeX("Automatic $p_0(x_g | e)$"),
       x=TeX("$p_{g}(x_g)$")) +
  theme(legend.position=c(.3, .3)) +
  ggtitle(TeX("C. Automatic policy after e"))

ggsave("plots/fp_after.pdf", height=4, width=3)

dfl %>%
  ggplot(aes(x=`p0(x|??)`, y=`p_{g_x}(??)`, color=DR)) +
  geom_point() +
  geom_line() +
  theme_classic() +
  labs(color=TeX("$\\Delta R_{g_x}$"),
       x=TeX("Automatic $p_0(x | e)$"),
       y=TeX("Controlled $p_{g_x}(e)$")) +
  theme(legend.position=c(.3, .4))

# CORRECTIONS AND FALSE STARTS

dfs = read_csv("output/stutter.csv") %>%
  select(-`...1`)

dfs %>%
  filter(alpha==1) %>%
  gather(measure, value, -alpha, -gamma) %>%
  filter(measure %in% c("healthy_corr", "pathological_corr")) %>%
  ggplot(aes(x=gamma, y=value, color=measure)) +
    geom_line(alpha=.7, size=3) +
    theme_classic() +
    labs(x=TeX("Gain $\\gamma$"), 
         y=TeX("Probability"),
         color="") +
    scale_color_hue(labels=c(TeX("Correction $p_g(c | x_{\\neq g})$"), 
                             TeX("False start $p_g(c | x_g)$"))) +
    theme(legend.position=c(.65, .5)) +
    ggtitle("D. Corrections and false starts")

ggsave("plots/corr_falsestart.pdf", width=3, height=4)


dfs %>%
  
