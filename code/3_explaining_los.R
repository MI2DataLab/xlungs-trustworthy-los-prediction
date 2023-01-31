#-----------------------------------------------------------------------------------
#:# DISCLAIMER: This code serves as an example of performing the described analysis.
#:# tlos_v1.csv is less precise (due to anonymization) than the very similar dataset
#:# used in original experiments, which will slightly impact the final results.
#-----------------------------------------------------------------------------------


df <- read.csv("../data/tlos_v1.csv")
dim(df)
colnames(df)

#:# data/feature subsets
library(dplyr)
library(ggplot2)
library(patchwork)

hist(df$time)

df_model <- df %>% mutate(time = log(1 + time))

hist(df_model$time)

df_risk <- df_model %>% select(time, outcome, sex, age)
dim(df_risk)
df_risk_annot <- df_model %>% select(!starts_with("pr"))
dim(df_risk_annot)
df_risk_pyrad <- df_model %>% select(!starts_with("an"))
dim(df_risk_pyrad)
df_risk_annot_pyrad <- df_model
dim(df_risk_annot_pyrad)

#:# classic approach
library(survival)

model_risk <- coxph(Surv(time, outcome) ~., df_risk)
model_risk_annot <- coxph(Surv(time, outcome) ~., df_risk_annot)
model_risk_pyrad <- coxph(Surv(time, outcome) ~., df_risk_pyrad)
model_risk_annot_pyrad <- coxph(Surv(time, outcome) ~., df_risk_annot_pyrad)

#:# model-specific interpretability
library(broom)

glance(model_risk)
augment(model_risk, df_risk)

tidy(model_risk)
tidy(model_risk) %>% filter(p.value < 0.05)
tidy(model_risk_annot) %>% filter(p.value < 0.05)
tidy(model_risk_pyrad) %>% filter(p.value < 0.05)
tidy(model_risk_annot_pyrad) %>% filter(p.value < 0.05)

library(xtable)
xt_model_risk <- xtable(tidy(model_risk) %>%
                          filter(p.value < 0.05) %>%
                          mutate(variable=term, .before = 1) %>%
                          select(-statistic, -term),
       caption = "only significant variables in the CoxPH model trained on age and sex",
       digits = 3)
print(xt_model_risk, include.rownames=FALSE, booktabs=TRUE)

xt_model_risk_annot <- xtable(tidy(model_risk_annot) %>%
                          filter(p.value < 0.05) %>%
                          mutate(variable=term, .before = 1) %>%
                          select(-statistic, -term),
                        caption = "only significant variables in the CoxPH model trained on age, sex and 17 human-annotated features",
                        digits = 3)
print(xt_model_risk_annot, include.rownames=FALSE, booktabs=TRUE)

xt_model_risk_pyrad <- xtable(tidy(model_risk_pyrad) %>%
                                filter(p.value < 0.05) %>%
                                mutate(variable=term, .before = 1) %>%
                                select(-statistic, -term),
                              caption = "only significant variables in the CoxPH model trained on age, sex and 76 algorithm-extracted features",
                              digits = 3)
print(xt_model_risk_pyrad, include.rownames=FALSE, booktabs=TRUE)

xt_model_risk_anot_pyrad <- xtable(tidy(model_risk_annot_pyrad) %>%
                                filter(p.value < 0.05) %>%
                                mutate(variable=term, .before = 1) %>%
                                select(-statistic, -term),
                              caption = "only significant variables in the CoxPH model trained on all 96 features",
                              digits = 3)
print(xt_model_risk_anot_pyrad, include.rownames=FALSE, booktabs=TRUE)

#:# use mlr3 because manually setting predict functions for blackboost is hard
library(mlr3proba)
library(mlr3extralearners)
library(mlr3pipelines)

task_risk <- as_task_surv(df_risk, id = "risk", time = "time", event = "outcome", type = "right")
task_risk_annot <- as_task_surv(df_risk_annot, id = "risk + annot", time = "time", event = "outcome", type = "right")
task_risk_pyrad <- as_task_surv(df_risk_pyrad, id = "risk + pyrad", time = "time", event = "outcome", type = "right")
task_risk_annot_pyrad <- as_task_surv(df_risk_annot_pyrad, id = "risk + annot + pyrad", time = "time", event = "outcome", type = "right")

learner_wb_risk <- lrn("surv.coxph", label="CPH")
learner_wb_risk_annot <- lrn("surv.coxph", label="CPH")
learner_wb_risk_pyrad <- lrn("surv.coxph", label="CPH")
learner_wb_risk_annot_pyrad <- lrn("surv.coxph", label="CPH")

learner_bb_risk <- lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=2000)
set.seed(0)
learner_bb_risk$train(task_risk)

learner_bb_risk_annot <- lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=2000)
set.seed(0)
learner_bb_risk_annot$train(task_risk_annot)

learner_bb_risk_annot_pyrad <- lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=2000)
set.seed(0)
learner_bb_risk_annot_pyrad$train(task_risk_annot_pyrad)

learner_bb_risk_pyrad <- lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=2000)
set.seed(0)
learner_bb_risk_pyrad$train(task_risk_pyrad)

learner_list <- list(
  learner_bb_risk,
  learner_bb_risk_annot,
  learner_bb_risk_annot_pyrad,
  learner_bb_risk_pyrad
)

#-- checkpoint
saveRDS(learner_list, paste0("../results/learner_list.rds"))
learner_list <- readRDS(paste0("../results/learner_list.rds"))

learner_bb_risk <- learner_list[[1]]
learner_bb_risk_annot <- learner_list[[2]]
learner_bb_risk_annot_pyrad <- learner_list[[3]]
learner_bb_risk_pyrad <- learner_list[[4]]
#--


#:# model-agnostic explainability approach
library(survex)

explainer_bb_risk <- explain(
  learner_bb_risk,
  data = df_risk[, -c(1, 2)],
  y = Surv(df_risk$time, df_risk$outcome)
)
explainer_bb_risk_annot <- explain(
  learner_bb_risk_annot,
  data = df_risk_annot[, -c(1, 2)],
  y = Surv(df_risk_annot$time, df_risk_annot$outcome)
)
explainer_bb_risk_annot_pyrad <- explain(
  learner_bb_risk_annot_pyrad,
  data = df_risk_annot_pyrad[, -c(1, 2)],
  y = Surv(df_risk_annot_pyrad$time, df_risk_annot_pyrad$outcome)
)
explainer_bb_risk_pyrad <- explain(
  learner_bb_risk_pyrad,
  data = df_risk_annot_pyrad[, -c(1, 2)],
  y = Surv(df_risk_annot_pyrad$time, df_risk_annot_pyrad$outcome)
)

explainer_bb_risk$times <- explainer_bb_risk_annot$times <-
  explainer_bb_risk_annot_pyrad$times <- explainer_bb_risk_pyrad$times <-
  head(explainer_bb_risk$times, -1)

set.seed(0)
vi_bb_risk <- model_parts(explainer_bb_risk, type="difference", N=NULL, B=15)
vi_bb_risk$result$times <- exp(vi_bb_risk$result$times)-1
plot(vi_bb_risk)

set.seed(0)
vi_bb_risk_annot <- model_parts(explainer_bb_risk_annot, type="difference", N=NULL, B=15)
vi_bb_risk_annot$result$times <- exp(vi_bb_risk_annot$result$times)-1
plot(vi_bb_risk_annot)

set.seed(0)
vi_bb_risk_pyrad <- model_parts(explainer_bb_risk_pyrad, type="difference", N=NULL, B=15)
vi_bb_risk_pyrad$result$times <- exp(vi_bb_risk_pyrad$result$times)-1
plot(vi_bb_risk_pyrad)

set.seed(0)
vi_bb_risk_annot_pyrad <- model_parts(explainer_bb_risk_annot_pyrad, type="difference", N=NULL, B=15)
vi_bb_risk_annot_pyrad$result$times <- exp(vi_bb_risk_annot_pyrad$result$times)-1
plot(vi_bb_risk_annot_pyrad)

vi_bb_list <- list(
  vi_bb_risk,
  vi_bb_risk_annot,
  vi_bb_risk_annot_pyrad,
  vi_bb_risk_pyrad
)

#-- checkpoint
saveRDS(vi_bb_list, paste0("../results/variable_importance_list.rds"))
vi_bb_list <- readRDS(paste0("../results/variable_importance_list.rds"))

vi_bb_risk <- vi_bb_list[[1]]
vi_bb_risk_annot <- vi_bb_list[[2]]
vi_bb_risk_annot_pyrad <- vi_bb_list[[3]]
vi_bb_risk_pyrad <- vi_bb_list[[4]]
#--


#----------------------------------------------------------------
#-- for a visual example: explain a smaller model using less data
#-- it doesn't significantly change results in practice
#-- while making the analysis considerably more accessible
#----------------------------------------------------------------

learner_bb_risk_annot <- lrn("surv.blackboost", label="GBDT", nu=0.01, mstop=500)
set.seed(0)
learner_bb_risk_annot$train(task_risk_annot)
set.seed(0)
id_seed0 <- sample(nrow(df_risk_annot), 150)

df_subset <- df_risk_annot[id_seed0, ]
EXPLAINER <- explain(
  learner_bb_risk_annot,
  data = df_subset[, -c(1, 2)],
  y = Surv(df_subset$time, df_subset$outcome)
)
EXPLAINER$times <- head(EXPLAINER$times, -1)

pp_list <- list()
for (i in 1:nrow(EXPLAINER$data)) { # this will take about 8h on a PC
  print(i)
  set.seed(0)
  pp_list[[i]] <- predict_parts(EXPLAINER, EXPLAINER$data[i,], p_max=8)  
}

#-- checkpoint
saveRDS(pp_list, paste0("../results/survshap_pmax=8_n=150_list.rds"))
pp_list <- readRDS(paste0("../results/survshap_pmax=8_n=150_list.rds"))
#--

JOINT_SURVSHAP_FI_COLORS <- DALEX::colors_discrete_drwhy(6)


#:# global explanation

integral_bytime <- function(x) {
  tmp <- (x[1:(length(x)-1)] + x[2:length(x)]) / 2
  sum(tmp)
}

lvi_list <- list()
tlvi_list <- list()
for (i in 1:length(pp_list)) {
  tlvi_list[[i]] <- abs(pp_list[[i]]$result)
  lvi_list[[i]] <- apply(pp_list[[i]]$result, 2, function(col) integral_bytime(abs(col)))
}
gvi <- rowMeans(do.call(cbind, lvi_list))
tgvi <- Reduce(`+`, tlvi_list) / length(tlvi_list)
dim(tgvi)

set.seed(0)
vi_bb_risk_annot_small <- model_parts(EXPLAINER, type="difference", N=NULL, B=15)
vi_bb_risk_annot_small$result$times <- exp(vi_bb_risk_annot_small$result$times)-1

temp <- vi_bb_risk_annot_small$result
temp <- temp[temp$permutation==0, ]
dim(temp)

temp[, 3:(ncol(temp)-3)] <- tgvi
vi_new <- vi_bb_risk_annot_small  
vi_new$result <- temp

top_variables <- names(tail(sort(gvi), 6))

df_plot_vi_wide <- vi_new$result[, c("times", top_variables)] 
colnames(df_plot_vi_wide)[2:7] <- stringr::str_to_sentence(
  stringr::str_replace_all(
    stringr::str_remove(
      colnames(df_plot_vi_wide)[2:7], "an_"), 
    "_", " ") 
)
df_plot_vi_long <- tidyr::pivot_longer(df_plot_vi_wide, !times)

p_tvi <- ggplot(data = df_plot_vi_long, 
       aes(x = times, y = value, color = name)) +
  geom_line(linewidth = 0.8, size = 0.8) +
  ylab("Aggregated |SurvSHAP(t) value|") + 
  xlim(0, 65) +
  xlab("Days since X-ray examination") +
  labs(title = "Feature importance in time", 
       subtitle = "Top 6 most important features in the GBDT model") +
  scale_color_manual(NULL, values = DALEX::colors_discrete_drwhy(6)[c(1, 2, 3, 4, 5, 6)]) +
  DALEX::theme_drwhy()

p_tvi


#-- what if?
ve_challenger_cat <- model_profile(EXPLAINER, variables = "an_medical_devices", 
                                   categorical_variables = "an_medical_devices",
                                   variable_splits = list("an_medical_devices"=c(0, 1)))
ve_challenger_cat$result[['_x_']] <- ifelse(as.numeric(as.character(ve_challenger_cat$result[['_x_']])) == 1, "Occurs", "Absent")
ve_challenger_cat$result[['_times_']] <- exp(ve_challenger_cat$result[['_times_']])-1
p_ve <- ggplot(data = ve_challenger_cat$result,
                              aes(x=`_times_`, 
                                  y=`_yhat_`,                         group = `_x_`,
                                  color = `_x_`)) +
  geom_line(linewidth = 0.8, size = 0.8) +
  guides(color = guide_legend(title="Value of Medical devices", 
                              override.aes = list(linewidth = 5))) +
  labs(title = "Partial dependence in time",
       subtitle = "For the most important feature", 
       y="Expected prediction",
       x="Days since X-ray examination") +
  xlim(0, 65) +
  scale_color_manual(NULL, values = DALEX::colors_discrete_drwhy(3)[c(3, 2)]) +
  DALEX::theme_drwhy()

p_exp1 <- p_tvi + p_ve
p_exp1
ggsave(p_exp1, filename = "../results/global_explanations.png", width = 11, height = 5)


#:# local explanation

#-- choose patient id for SurvSHAP(t)
s_id <- 1 # 1..150
s_survshap <- pp_list[[s_id]]
plot(s_survshap)
s_lvi <- lvi_list[[s_id]]
s_top_variables <- names(tail(sort(s_lvi), 6))

copy_s_survshap <- s_survshap
copy_s_survshap$eval_times <- exp(copy_s_survshap$eval_times) - 1
copy_s_survshap$result <- as.data.frame(copy_s_survshap$result[, s_top_variables])
top_variable_values <- unlist(EXPLAINER$data[s_id, s_top_variables])
top_variable_values_nice <- ifelse(top_variable_values == 0, "Absent", 
                                   ifelse(top_variable_values == 1, "Occurs",
                                          ifelse(top_variable_values == 0.5, "Maybe occurs", top_variable_values)))

# fix this value
if ('sex' %in% names(top_variable_values)) top_variable_values_nice['sex'] <- ifelse(top_variable_values_nice['sex'] == "Absent", "Female", "Male")

colnames(copy_s_survshap$result)

colnames(copy_s_survshap$result) <- paste0(
  stringr::str_to_sentence(
    stringr::str_replace_all(
      stringr::str_remove(
        colnames(copy_s_survshap$result), "an_"), 
      "_", " ") 
  ),
  " = ",
  top_variable_values_nice)

colnames(copy_s_survshap$result) 

df_plot_survshap_wide <- cbind(copy_s_survshap$result, times=copy_s_survshap$eval_times)
df_plot_survshap_long <- tidyr::pivot_longer(df_plot_survshap_wide, !times)

p_tsv <- ggplot(data = df_plot_survshap_long, 
                aes(x = times, y = value, color = name)) +
  geom_line(linewidth = 0.8, size = 0.8) +
  ylab("SurvSHAP(t) value") + 
  xlim(0, 65) +
  xlab("Days since X-ray examination") +
  labs(title = "SurvSHAP(t) for a selected patient", 
       subtitle = "Top 6 most important features for this prediction") +
  scale_color_manual(NULL,
                     values = DALEX::colors_discrete_drwhy(6)[c(1, 4, 2, 3, 5, 6)]) +
  DALEX::theme_drwhy()
p_tsv


#-- what if?
cp_variable_of_interest <- "an_pleural_effusion"
cp_variable_of_interest_nice <- "Pleural effusion"
cp <- predict_profile(EXPLAINER, EXPLAINER$data[s_id,],
                      variables = cp_variable_of_interest, 
                      categorical_variables = cp_variable_of_interest)
cp$result[[cp_variable_of_interest]] <- ifelse(cp$result[[cp_variable_of_interest]] == 1, "Occurs", 
                                               ifelse(cp$result[[cp_variable_of_interest]] == 0.5, "Maybe occurs",
                                               "Absent"))
cp$result[['_times_']] <- exp(cp$result[['_times_']])-1

p_cp <- ggplot(data = cp$result,
               aes(x=`_times_`, 
                   y=`_yhat_`,
                   color = an_pleural_effusion)) +
  geom_line(linewidth = 0.8, size = 0.8) +
  guides(color = guide_legend(title=paste0("Value of ", cp_variable_of_interest_nice), 
                              override.aes = list(linewidth = 5))) +
  labs(title = "What-if analysis for a selected patient",
       subtitle = "For the selected ambiguous feature", 
       y="Prediction",
       x="Days since X-ray examination") +
  xlim(0, 65) +
  scale_color_manual(NULL, values = DALEX::colors_discrete_drwhy(3)[c(3,1,2)]) +
  DALEX::theme_drwhy()
p_cp

p_exp2 <- p_tsv + p_cp
p_exp2
ggsave(p_exp2, filename = "../results/local_explanations.png", width = 11, height = 5)
