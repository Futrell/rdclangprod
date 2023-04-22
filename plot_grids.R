rm(list=ls())
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
  ylim(0, 1) +
  labs(x=TeX("Control Gain $\\alpha$"), 
       y=TeX("Future Discount $\\gamma$"), 
       color="") +
  theme(legend.position="right",
        legend.title=element_text(size=9),
        legend.margin=margin(0,0,0,0),
        legend.box.margin=margin(-5,-5,-5,-5)) +
  scale_color_viridis(guide = guide_colorbar(title.position="bottom", 
                                             label.position="right"))
                                             #barwidth=7,
                                             #barheight=.5))
  
ggsave("plots/shortlong.pdf", width=3.75, height=2)


# FILLED PAUSE

dfl_policy = read_csv("output/fp_policy.csv") %>%
  select(-`...1`) %>%
  mutate(p_L=exp(R_g - log(5)))

dfl_policy %>%
  ggplot(aes(x=factor(x, levels=c("5", "4", "3", "2", "1", "0")),
             y=as.factor(g), 
             fill=R_g)) +
  geom_tile(color="black") +
  scale_x_discrete(labels=c(
    TeX("$a_5$"),
    TeX("$a_4$"), 
    TeX("$a_3$"), 
    TeX("$a_2$"),
    TeX("$a_1$"), 
    TeX("$e$")
  )) +
  #scale_fill_gradient2(low="yellow", high="blue", midpoint=0) +
  scale_fill_viridis() +
  coord_flip() +
  labs(x="Action a", y="Goal g", fill=TeX("$R_g(a)$")) +
  guides(color="none") +
  theme_classic() +
  theme(line=element_blank()) +
  ggtitle(TeX("A. Reward with filled pause e"))

ggsave("plots/fp_reward.pdf", height=4, width=3.5)

dfl = read_csv("output/fp_summary.csv") %>%
  select(-`...1`) %>%
  mutate(g=1:nrow(.)) 

dfl %>%
  ggplot(aes(x=g, y=`p_{g_x}(ε)`, color=`p_{g_x}(x)`)) +
    geom_line(size=2) +
    geom_point(size=3) +
    theme_classic() +
    labs(x=TeX("Goal $g$"),
         y=TeX("Filled pause probability $\\pi_g(e)$"),
         color=TeX("$\\pi_g(a_g)$")) +
    theme(legend.position=c(.8, .7)) +
    scale_color_viridis() +
  
    ggtitle("B. Filled pause probability")

ggsave("plots/fp_prob.pdf", height=4, width=3)



dfl %>%
  ggplot(aes(x=factor(g), color=`p_{g_x}(x)`, y=`p0(x|ε)`, group=factor(1))) +
  geom_point(size=3) +
  geom_line(size=2) +
  theme_classic() +
  scale_color_viridis() +
  labs(color=TeX("$\\pi_g(a_g)$"),
       y=TeX("Automatic probability after filled pause $\\pi_0(a | e)$"),
       x=TeX("Action $a$")) +
  scale_x_discrete(labels=c(
    TeX("$a_1$"),
    TeX("$a_2$"), 
    TeX("$a_3$"), 
    TeX("$a_4$"),
    TeX("$a_5$") 
  )) +  
  theme(legend.position=c(.8, .7)) +
  ggtitle(TeX("C. Automatic policy after e"))

ggsave("plots/fp_after.pdf", height=4, width=3)

# CORRECTIONS AND FALSE STARTS

dfs = read_csv("output/stutter.csv") %>%
  select(-`...1`)

dfs %>%
  filter(alpha==1) %>%
  gather(measure, value, -alpha, -gamma) %>%
  filter(measure %in% c("healthy_corr", "pathological_corr")) %>%
  ggplot(aes(x=gamma, y=value, color=measure)) +
    geom_line(alpha=.7, size=2) +
    theme_classic() +
    xlim(1, NA) +
    labs(x=TeX("Control gain $\\alpha$"), 
         y=TeX("Probability of correction"),
         color="") +
    scale_color_hue(labels=c(TeX("After error $\\pi_g(c | a_{\\neq g})$"), 
                             TeX("False start $\\pi_g(c | a_g)$"))) +
    theme(legend.position=c(.65, .5)) +
    ggtitle("D. Corrections and false starts")

ggsave("plots/corr_falsestart.pdf", width=3, height=4)

  
