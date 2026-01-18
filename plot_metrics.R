library(ggplot2)
library(tidyverse)

library(pheatmap)

df = read.csv("MFO_metrics_df.csv")
long_df <- df %>%
  pivot_longer(
    cols = -Configuration,
    names_to = "score_type",
    values_to = "value"
  )
score_mat_df <- long_df %>%
  pivot_wider(
    names_from = Configuration,
    values_from = value
  )%>%
  filter(score_type %in% c("F.max", "S.min...F.max", "Overfitting_Score")) %>%
  column_to_rownames("score_type") %>%
  as.matrix()

annotation_col <- df %>% select(Configuration) %>%
  extract(
    Configuration,
    into = c("learning_rate", "architecture", "epochs"),
    regex = "LR-(.+)_ARCH-\\[([0-9, ]+)\\]_EPOCHS-([0-9]+)",
    convert = FALSE, remove=FALSE
  )%>%
  column_to_rownames("Configuration")

  
pheatmap(score_mat_df,scale="row", border=FALSE, show_colnames = FALSE,cluster_rows=FALSE, cluster_cols=FALSE, annotation_col = annotation_col)

ggplot(long_df %>% 
         filter(score_type %in% c("F.max", "S.min...F.max", "Overfitting_Score")) %>%
         extract(
  Configuration,
  into = c("learning_rate", "architecture", "epochs"),
  regex = "LR-(.+)_ARCH-\\[([0-9, ]+)\\]_EPOCHS-([0-9]+)",
  convert = FALSE, remove=FALSE), aes(x=Configuration, y=value, fill=learning_rate)) +
  geom_col() +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_wrap(~score_type, scale="free", nrow=3)


ggplot(long_df %>% 
         filter(score_type %in% c("F.max", "S.min...F.max", "Overfitting_Score")) %>%
         extract(
           Configuration,
           into = c("learning_rate", "architecture", "epochs"),
           regex = "LR-(.+)_ARCH-\\[([0-9, ]+)\\]_EPOCHS-([0-9]+)",
           convert = FALSE, remove=FALSE), aes(x=Configuration, y=value, fill=architecture)) +
  geom_col() +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_wrap(~score_type, scale="free", nrow=3)


library(ggh4x)
ggplot(long_df %>% 
         filter(score_type %in% c("F.max", "S.min...F.max", "Overfitting_Score")) %>%
         extract(
           Configuration,
           into = c("learning_rate", "architecture", "epochs"),
           regex = "LR-(.+)_ARCH-\\[([0-9, ]+)\\]_EPOCHS-([0-9]+)",
           convert = FALSE, remove=FALSE) %>%
         mutate(architecture = factor(architecture, levels= c('256', '256, 128', '256, 128, 64', '256, 128, 62, 32'))) %>%
         mutate(learning_rate = factor(learning_rate, levels = c("0.01", "0.001", "0.0001", "1e-05")))
         , aes(x=interaction(epochs, learning_rate), y=value)) %>% +
  geom_rect(aes(ymin=-20, ymax=35, width=1, fill=learning_rate), alpha=0.5) +
  geom_line(aes(group=learning_rate), linewidth=0.25) +
  geom_point(aes(color=epochs), size=1.5) +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_wrap(~score_type+architecture, scale="free_y", nrow=3) + 
  facetted_pos_scales(y=list(
    scale_y_continuous(limits = c(0,1)),
    scale_y_continuous(limits = c(0,1)),
    scale_y_continuous(limits = c(0,1)),
    scale_y_continuous(limits = c(0,1)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-1, 30)),
    scale_y_continuous(limits = c(-1, 30)),
    scale_y_continuous(limits = c(-1, 30)),
    scale_y_continuous(limits = c(-1, 30))
  )) + scale_fill_manual(name="hparams", values=c(
    "0.0001"="cornflowerblue", 
    "0.001"="blue",
    "0.01" = "navy",
    "1e-05"="skyblue")) + 
      scale_color_manual(values=c( 
        "10" = "lightpink",
        "15" = "salmon",
        "20" = "tomato",
        "30" = "red",
        "40" = "darkred"))


ggplot(long_df %>% 
         filter(score_type %in% c("F.max", "S.min...F.max", "Overfitting_Score")) %>%
         extract(
           Configuration,
           into = c("learning_rate", "architecture", "epochs"),
           regex = "LR-(.+)_ARCH-\\[([0-9, ]+)\\]_EPOCHS-([0-9]+)",
           convert = FALSE, remove=FALSE) %>%
         mutate(architecture = factor(architecture, levels= c('256', '256, 128', '256, 128, 64', '256, 128, 62, 32'))) %>%
         mutate(learning_rate = factor(learning_rate, levels = c("0.01", "0.001", "0.0001", "1e-05")))
       , aes(x=epochs, y=value)) %>% +
  geom_hline(yintercept=0, color="gray20") +
  geom_line(aes(group=learning_rate, color=learning_rate), linewidth=.5) +
  geom_point(aes(color=learning_rate), size=1) +
  theme_bw() +
  facet_wrap(~score_type+architecture, scale="free_y", nrow=3) + 
  facetted_pos_scales(y=list(
    scale_y_continuous(limits = c(0,.8)),
    scale_y_continuous(limits = c(0,.8)),
    scale_y_continuous(limits = c(0,.8)),
    scale_y_continuous(limits = c(0,.8)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-0.125, 0.05)),
    scale_y_continuous(limits = c(-1, 28)),
    scale_y_continuous(limits = c(-1, 28)),
    scale_y_continuous(limits = c(-1, 28)),
    scale_y_continuous(limits = c(-1, 28))
  )) +
  scale_color_manual(name="Learning Rate", values=c(
    "0.0001"="cornflowerblue", 
    "0.001"="royalblue3",
    "0.01" = "midnightblue",
    "1e-05"="skyblue1"))
