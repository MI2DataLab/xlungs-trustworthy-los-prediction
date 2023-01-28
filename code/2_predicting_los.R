df <- read.csv("../data/tlos_v1.csv")
dim(df)
colnames(df)

library(dplyr)

hist(df$time)

df_model <- df %>% mutate(time = log(1 + time))

hist(df_model$time)

df_risk <- df_model %>% select(time, outcome, sex, age)
df_risk_annot <- df_model %>% select(!starts_with("pr"))
df_risk_pyrad <- df_model %>% select(!starts_with("an"))
df_risk_annot_pyrad <- df_model

library(mlr3proba)
library(mlr3extralearners)
library(mlr3pipelines)

task_risk <- as_task_surv(
  df_risk, 
  id = "risk", time = "time", 
  event = "outcome", type = "right"
)
task_risk_annot <- as_task_surv(
  df_risk_annot, 
  id = "risk + annot", time = "time", 
  event = "outcome", type = "right"
)
task_risk_pyrad <- as_task_surv(
  df_risk_pyrad, id = "risk + pyrad", time = "time", 
  event = "outcome", type = "right"
)
task_risk_annot_pyrad <- as_task_surv(
  df_risk_annot_pyrad, id = "risk + annot + pyrad", time = "time", 
  event = "outcome", type = "right"
)

tasks <- c(
  task_risk, 
  task_risk_annot,
  task_risk_pyrad,
  task_risk_annot_pyrad
)

learners <-  c(
  lrn("surv.coxph", label="CPH"), #fallback=lrn("surv.kaplan")), # if cph fails on large datasets
  
  lrn("surv.rfsrc", label="RSF", splitrule="bs.gradient"),
  lrn("surv.ranger"),
  lrn("surv.ctree"),
  
  # defaults are too restrictive on time efficiency
  lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=2000),
  
  as_learner(
    po("scale") %>>%
      po("learner", 
         learner=lrn("surv.deepsurv", 
                     epochs=1000, 
                     optimizer = "adadelta", 
                     num_nodes=c(8), 
                     batch_size=64, 
                     lr=0.1)
      )
  ), 
  as_learner(
    po("scale") %>>%
      po("learner", 
         learner=lrn("surv.deephit", 
                     epochs=1000, 
                     optimizer = "adadelta", 
                     num_nodes=c(8), 
                     batch_size=64, 
                     lr=1)
      )
  )
)

nrepeats <- 10
nfolds <- 10

resamplings <- rsmp("repeated_cv", repeats=nrepeats, folds=nfolds)

grid <- benchmark_grid(tasks, learners, resamplings)

set.seed(0)

## retrieve log from error
#bmr <- mlr3misc::encapsulate("callr", benchmark, list(design=grid))

## parallel computing
future::plan("multisession", workers=4)
bmr <- benchmark(grid)
bmr

#-- checkpoint
saveRDS(bmr, paste0("../results/benchmark_", nrepeats, "x", nfolds, ".rds"))
bmr <- readRDS(paste0("../results/benchmark_", nrepeats, "x", nfolds, ".rds"))
#--

scores_cix <- bmr$score(msr("surv.cindex", tiex=0.5))
# integrated brier score
scores_ibs <- bmr$score(msr("surv.graf", integrated=T, method=2))


colnames(scores_ibs)[ncol(scores_ibs)] <- "score"
scores_ibs$score_name <- "Integrated Brier Score (lower is better)"
colnames(scores_cix)[ncol(scores_cix)] <- "score"
scores_cix$score_name <- "C-Index (higher is better)"

scores <- rbind(scores_cix, scores_ibs)

scores$learner_id[scores$learner_id == "surv.coxph"] <- "CoxPH"
scores$learner_id[scores$learner_id == "surv.rfsrc"] <- "RF-SRC"
scores$learner_id[scores$learner_id == "surv.ranger"] <- "Ranger"
scores$learner_id[scores$learner_id == "surv.ctree"] <- "CTree"
scores$learner_id[scores$learner_id == "scale.surv.deepsurv"] <- "DeepSurv"
scores$learner_id[scores$learner_id == "scale.surv.deephit"] <- "DeepHit"
scores$learner_id[scores$learner_id == "surv.blackboost"] <- "GBDT"

scores$learner_id <- factor(scores$learner_id,
                            levels=c("DeepHit", "DeepSurv", "Ranger",
                                     "CTree", "CoxPH", "RF-SRC", "GBDT"))
scores$task_id <- factor(scores$task_id,
                         levels = c("risk", "risk + pyrad", "risk + annot", "risk + annot + pyrad"))

#-- checkpoint
saveRDS(scores %>% select(task_id, learner_id, resampling_id, score, score_name, iteration),
        paste0("../results/scores_", nrepeats, "x", nfolds, ".rds"))
scores <- readRDS(paste0("../results/scores_", nrepeats, "x", nfolds, ".rds"))
#--

library(ggplot2)
library(ggstatsplot)
library(patchwork)

p_comparison <- grouped_ggbetweenstats(
  data             = scores %>% filter(
    task_id == "risk + annot + pyrad",
    learner_id != "DeepHit", # it does not converge,
    score > 0.5 | score < 0.22
  ),
  x                = learner_id,
  y                = score,
  grouping.var     = score_name,
  results.subtitle = FALSE,
  k                = 3,
  pairwise.comparisons = FALSE,
  bf.message       = FALSE,
  p.adjust.method  = "bonferroni",
  xlab             = "Survival model", 
  ylab             = "Measure value",
  annotation.args  = list(title=paste0("Model benchmark using all features [",
                                       nrepeats,
                                       " repeats of ",
                                       nfolds,
                                       "-fold cross-validation]"),
                          caption="Note: Most of the DeepHit models did not converge and provide random predictions; thus, DeepHit is removed from the comparison."
  ),
  # remove second y axis https://github.com/IndrajeetPatil/ggstatsplot/blob/2a7ee97822f39e7f2fce6d6f3d8a2e5ec9662f3f/R/ggbetweenstats_helpers.R#L244
  ggplot.component = list(
    scale_y_continuous(),
    scale_x_discrete(),
    scale_colour_manual(values=rep("darkgrey", 6))
  ),
  point.args = list(
    position = ggplot2::position_jitterdodge(dodge.width = 0.7), 
    alpha = 0.4, size = 2, stroke = 0
  )
) & theme(axis.text=element_text(size=9))

p_comparison

ggsave(p_comparison, filename = "../results/model_comparison.png", width = 10, height = 4)

p_bb <- grouped_ggbetweenstats(
  data             = scores %>% filter(learner_id == "GBDT"),
  x                = task_id,
  y                = score,
  grouping.var     = score_name,
  results.subtitle = FALSE,
  k                = 3,
  # pairwise.display = "all",
  bf.message = FALSE,
  p.adjust.method = "bonferroni",
  xlab="Feature set", ylab="Measure value",
  annotation.args = list(title=paste0("GBDT (black-box) performance on different feature sets [", 
                                      nrepeats,
                                      " repeats of ", 
                                      nfolds, 
                                      "-fold cross-validation]")),
  ggplot.component = list(
    scale_y_continuous(),
    scale_x_discrete(labels=c("baseline\n(d=2)", "algorithm-extracted\n(d=2+76)", "human-annotated\n(d=2+17)", "all features\n(d=2+17+76)")),
    scale_colour_manual(values=rep("grey", 6))
  ),
  point.args = list(
    position = ggplot2::position_jitterdodge(dodge.width = 0.7), 
    alpha = 0.4, size = 2, stroke = 0
  )
) & theme(axis.text=element_text(size=8.75))

p_bb

ggsave(p_bb, filename = "../results/bb_performance.png", width = 10, height = 5)


p_wb <- grouped_ggbetweenstats(
  data             = scores %>% filter(learner_id == "CoxPH"),
  x                = task_id,
  y                = score,
  grouping.var     = score_name,
  results.subtitle = FALSE,
  k                = 3,
  # pairwise.display = "all",
  bf.message = FALSE,
  p.adjust.method = "bonferroni",
  xlab="Feature set", ylab="Measure value",
  annotation.args = list(title=paste0("CoxPH (white-box) performance on different feature sets [", 
                                      nrepeats,
                                      " repeats of ", 
                                      nfolds, 
                                      "-fold cross-validation]")),
  ggplot.component = list(
    scale_y_continuous(),
    scale_x_discrete(labels=c("baseline\n(d=2)", "algorithm-extracted\n(d=2+76)", "human-annotated\n(d=2+17)", "all features\n(d=2+17+76)")),
    scale_colour_manual(values=rep("grey", 6))
  ),
  point.args = list(
    position = ggplot2::position_jitterdodge(dodge.width = 0.7), 
    alpha = 0.4, size = 2, stroke = 0
  )
) & theme(axis.text=element_text(size=8.75))

p_wb

ggsave(p_wb, filename = "../results/wb_performance.png", width = 10, height = 5)


# --- retrieve the p value for final model comparison

grouped_ggbetweenstats(
  data             = rbind(
    scores %>% filter(learner_id == "CoxPH", task_id == "risk + annot"),
    scores %>% filter(learner_id == "GBDT", task_id == "risk + annot + pyrad")
  ),
  x                = learner_id,
  y                = score,
  grouping.var     = score_name,
  k                = 3,
  bf.message       = FALSE,
  p.adjust.method  = "bonferroni",
  pairwise.display = "all"
)
