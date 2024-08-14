# This file is for formal simulation
# system("type R")
Sys.setenv("R_LIBS_USER"="/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2")
library(MASS, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(tidyverse, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(CBPS, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(foreach, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(doParallel, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(mvtnorm, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
library(MatchIt, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
source('/storage/work/yqs5519/TuneCalibratedLoss/algorithm/helper_train.R')

start_time = Sys.time()
print(start_time)
######################################################################## settings 
nsim = 1000
nmodel = 6

p = 4
n = 200
Mu = rep(0,p)
Correlation = diag(p)
# Correlation = CSgenerate(4, 0.5)
# Gamma = c(-1, 0.5, -0.25, -0.1) # treatment model

Gamma = c(-3, 0.5, -0.25, -0.1) # treatment model: review's comments
Beta1 = c(27.4, 13.7, 13.7, 13.7) # outcome model
Beta0 = c(27.4, 13.7, 13.7, 13.7) # outcome model
True_treated = 210
True_control = 200

ps_correct = 'no'
out_correct = 'yes'

file_num = readr::parse_number(list.files('/storage/work/yqs5519/TuneCalibratedLoss/', 'sim_results'))
location_index = ifelse(is_empty(file_num), 1,  max(file_num) + 1) ## set location to save results
location = paste('/storage/work/yqs5519/TuneCalibratedLoss/sim_results', location_index, '/', sep = '')
location2 = paste('/storage/work/yqs5519/TuneCalibratedLoss/sim_data', location_index, '/', sep = '')
dir.create(location)
dir.create(location2)


fileConn<-file(paste(location, 'settings.txt', sep = ''))
writeLines(c("Sample size:", n, '\n',
             "Gamma (treatment model):", Gamma, '\n',
             'Beat1 (outcome model):', Beta1, '\n',
             'Beat0 (outcome model):', Beta0, '\n',
             'Mean of X:', Mu, '\n',
             'Correlation of X:', Correlation, '\n',
             'True mean treated:', True_treated, '\n',
             'True mean control:', True_control, '\n',
             'Number of iteration:', nsim, '\n',
             'ps model correct:', ps_correct, '\n',
             'out model correct:', out_correct, '\n',
             # 'Binary:', "yes",'\n',
             'Output locaton:', location), fileConn)
close(fileConn)
######################################################################## Simulation

## simulate
multiResultClass <- function(result1=NULL,result2=NULL,result3=NULL,result4=NULL, result5=NULL, result6=NULL, result7=NULL, result8=NULL, result9=NULL)
{
  me <- list(
    ate_HT = result1,
    ate_Hajek = result2,
    ate_DR = result3,
    bias_ps = result4,
    balance_df = result5,
    best_tune_nn = result6,
    best_tune_glm = result7,
    balance_df_mis = result8,
    balance_df_true = result9
  )
  
  ## Set the name for the class
  class(me) <- append(class(me), "multiResultClass")
  return(me)
}

ate_HT=data.frame(matrix(ncol = nmodel, nrow = 0))
ate_Hajek=data.frame(matrix(ncol = nmodel, nrow = 0))
ate_DR=data.frame(matrix(ncol = nmodel, nrow = 0))
bias_ps=data.frame(matrix(ncol = nmodel-1, nrow = 0))
balance_df=list()
balance_df_mis=list()
balance_df_true=list()

num_cores <- detectCores()
cl <- makeCluster(num_cores)
registerDoParallel(cl)


res <- foreach(i = 1:nsim) %dopar%  {
  library(MASS, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(tidyverse, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(dplyr, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(CBPS, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(foreach, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(doParallel, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  library(mvtnorm, lib.loc = '/storage/home/yqs5519/R/x86_64-pc-linux-gnu-library/4.2')
  
  output <- multiResultClass()
  
  set.seed(i)

  # continuous X
  X<-MASS::mvrnorm(n,mu=Mu,Sigma=Correlation)
  prop<-1/(1+exp(-(X %*% Gamma))) %>% as.vector() # true propensity score
  treat<-rbinom(n,1,prop)
  y1<-True_treated+as.vector(X %*% Beta1)+rnorm(n)
  y0<-True_control+as.vector(X %*% Beta0)+rnorm(n)
  y<-treat*y1 + (1-treat)*y0

  mean_col_true <- apply(X, 2, mean)
  std_col_true <- apply(X, 2, sd)
  X_centered_true <- sweep(X, 2, mean_col_true, "-")
  X_standardized_true <- sweep(X_centered_true, 2, std_col_true, "/")
  X_true = X_standardized_true
  sim_data_true = data.frame(Y = y, treatment = treat, X = X_true)
  write.csv(sim_data_true, paste0(paste(location2, 'sim_data_true', sep = ''), i, ".csv"), row.names = F)

  X.mis<-cbind(exp(X[,1]/2),X[,2]*(1+exp(X[,1]))^(-1)+10, (X[,1]*X[,3]/25+.6)^3,(X[,2]+X[,4]+20)^2)
  mean_col <- apply(X.mis, 2, mean)
  std_col <- apply(X.mis, 2, sd)
  X_centered <- sweep(X.mis, 2, mean_col, "-")
  X_standardized <- sweep(X_centered, 2, std_col, "/")
  X.mis = X_standardized
  sim_data_mis = data.frame(Y = y, treatment = treat, X = X.mis)
  write.csv(sim_data_mis, paste0(paste(location2, 'sim_data_mis', sep = ''), i, ".csv"), row.names = F)
  
  # run Python script with argument
  my_argument = c(i, location2, 'NN_IS', ps_correct)
  system(paste("/storage/work/yqs5519/TuneCalibratedLoss/algorithm/rpython", paste(my_argument, collapse=" ")))

  my_argument2 = c(i, location2, 'GLM_IS', ps_correct)
  system(paste("/storage/work/yqs5519/TuneCalibratedLoss/algorithm/rpython", paste(my_argument2, collapse=" ")))

  my_argument3 = c(i, location2, 'NN_only', ps_correct)
  system(paste("/storage/work/yqs5519/TuneCalibratedLoss/algorithm/rpython", paste(my_argument3, collapse=" ")))

  ## model outcome
  if (out_correct == 'yes') {
    out_data = read.csv(paste0(paste(location2, 'sim_data_true', sep = ''), i, ".csv"))
  } 
  if (out_correct == 'no') {
    out_data = read.csv(paste0(paste(location2, 'sim_data_mis', sep = ''), i, ".csv"))
  }
  out_model = lm(Y~treatment+X.1+X.2+X.3+X.4, out_data)
  m1 <- predict(out_model, newdata = data.frame(treatment = 1,
                                                dplyr::select(out_data[,-1], !treatment)),
                type = "response")
  m0 <- predict(out_model, newdata = data.frame(treatment = 0,
                                                dplyr::select(out_data[,-1], !treatment)),
                type = "response")

  ## unadjusted
  PS_unadj = NULL

  ## logistics
  if (ps_correct == 'yes') {
    data = read.csv(paste0(paste(location2, 'sim_data_true', sep = ''), i, ".csv"))
    X.obs = as.matrix(data[,3:ncol(data)])
  } 
  if (ps_correct == 'no') {
    data = read.csv(paste0(paste(location2, 'sim_data_mis', sep = ''), i, ".csv"))
    X.obs = as.matrix(data[,3:ncol(data)])
  }
  model = glm(treatment ~ ., data = data[,-1], family = binomial(link = 'logit'))
  PS_logistic = model$fitted.values

  ## proposed --> tune: select the one with best balance
  ## proposed: NN IS
  PS_proposed_matrix = read.csv(paste0(paste(location2, 'NN_IS_PS_results_pred', sep = ''), i, ".csv"), header = T) 
  balance = c()
  for (p in 1:ncol(PS_proposed_matrix)) {
    balance[p] = sum(abs(weighted_mean_diff(X.obs, PS_proposed_matrix[,p], data$treatment)))
  }
  if (length(balance) == sum(is.na(balance))) {
    PS_NN_IS = rep(NaN, nrow(PS_proposed_matrix))
    balance_index = NaN} else {
      PS_NN_IS = PS_proposed_matrix[, which(balance == min(balance, na.rm = T))]
      balance_index = which(balance == min(balance, na.rm = T))
    }
  
  output$best_tune_nn = c(output$best_tune_nn, balance_index)
  
  ## proposed: GLM IS
  PS_proposed_matrix = read.csv(paste0(paste(location2, 'GLM_IS_PS_results_pred', sep = ''), i, ".csv"), header = T) 
  balance = c()
  for (p in 1:ncol(PS_proposed_matrix)) {
    balance[p] = sum(abs(weighted_mean_diff(X.obs, PS_proposed_matrix[,p], data$treatment)))
  }
  if (length(balance) == sum(is.na(balance))) {
    PS_GLM_IS = rep(NaN, nrow(PS_proposed_matrix))
    balance_index = NaN} else {
      PS_GLM_IS = PS_proposed_matrix[, which(balance == min(balance, na.rm = T))]
      balance_index = which(balance == min(balance, na.rm = T))
    }
  output$best_tune_glm = c(output$best_tune_glm, balance_index)
  
  ## proposed: NN
  PS_NN = read.csv(paste0(paste(location2, 'NN_PS', sep = ''), i, ".csv"), header = F) %>% as.matrix() %>% as.vector

  ## CBPS_exact
  fit_CBPS_exact = CBPS(treatment ~ ., data = data[,-1], method = 'exact', ATT=0)
  PS_CBPS = fit_CBPS_exact$fitted.values

  ## estimate effect
  ate_HT_vec = c(mean_ate_HT(data$treatment, data$Y, NULL),
                 mean_ate_HT(data$treatment, data$Y, PS_logistic),
                 mean_ate_HT(data$treatment, data$Y, PS_GLM_IS),
                 mean_ate_HT(data$treatment, data$Y, PS_NN),
                 mean_ate_HT(data$treatment, data$Y, PS_NN_IS),
                 mean_ate_HT(data$treatment, data$Y, PS_CBPS))

  ate_Hajek_vec = c(mean_ate_Hajek(data$treatment, data$Y, NULL),
                    mean_ate_Hajek(data$treatment, data$Y, PS_logistic),
                    mean_ate_Hajek(data$treatment, data$Y, PS_GLM_IS),
                    mean_ate_Hajek(data$treatment, data$Y, PS_NN),
                    mean_ate_Hajek(data$treatment, data$Y, PS_NN_IS),
                    mean_ate_Hajek(data$treatment, data$Y, PS_CBPS))

  ate_DR_vec = c(mean_ate_DR(data$treatment, data$Y, NULL, m1, m0),
                 mean_ate_DR(data$treatment, data$Y, PS_logistic, m1, m0),
                 mean_ate_DR(data$treatment, data$Y, PS_GLM_IS, m1, m0),
                 mean_ate_DR(data$treatment, data$Y, PS_NN, m1, m0),
                 mean_ate_DR(data$treatment, data$Y, PS_NN_IS, m1, m0),
                 mean_ate_DR(data$treatment, data$Y, PS_CBPS, m1, m0))
  output$ate_HT =  rbind(output$ate_HT, ate_HT_vec)
  output$ate_Hajek =  rbind(output$ate_Hajek, ate_Hajek_vec)
  output$ate_DR =  rbind(output$ate_DR, ate_DR_vec)


  ## bias of ps estimation
  bias_vec = c(mean(abs(PS_logistic - prop)),
               mean(abs(PS_GLM_IS - prop)),
               mean(abs(PS_NN - prop)),
               mean(abs(PS_NN_IS - prop)),
               mean(abs(PS_CBPS - prop)))
  output$bias_ps =  rbind(output$bias_ps, bias_vec)


  ## covariance balance
  cb = data.frame(unadjusted = weighted_mean_diff(X.obs, PS_unadj, data$treatment),
                          logistic = weighted_mean_diff(X.obs, PS_logistic, data$treatment),
                          GLM_IS = weighted_mean_diff(X.obs, PS_GLM_IS, data$treatment),
                          NNet = weighted_mean_diff(X.obs, PS_NN, data$treatment),
                          NNet_IS = weighted_mean_diff(X.obs, PS_NN_IS, data$treatment),
                          CBPS_exact = weighted_mean_diff(X.obs, PS_CBPS, data$treatment))

  output$balance_df = c(output$balance_df, list(cb))

  cb.true = data.frame(unadjusted = weighted_mean_diff(X_true, PS_unadj, data$treatment),
                  logistic = weighted_mean_diff(X_true, PS_logistic, data$treatment),
                  GLM_IS = weighted_mean_diff(X_true, PS_GLM_IS, data$treatment),
                  NNet = weighted_mean_diff(X_true, PS_NN, data$treatment),
                  NNet_IS = weighted_mean_diff(X_true, PS_NN_IS, data$treatment),
                  CBPS_exact = weighted_mean_diff(X_true, PS_CBPS, data$treatment))
  
  output$balance_df_true = c(output$balance_df_true, list(cb.true))
  
  cb.mis = data.frame(unadjusted = weighted_mean_diff(X.mis, PS_unadj, data$treatment),
                  logistic = weighted_mean_diff(X.mis, PS_logistic, data$treatment),
                  GLM_IS = weighted_mean_diff(X.mis, PS_GLM_IS, data$treatment),
                  NNet = weighted_mean_diff(X.mis, PS_NN, data$treatment),
                  NNet_IS = weighted_mean_diff(X.mis, PS_NN_IS, data$treatment),
                  CBPS_exact = weighted_mean_diff(X.mis, PS_CBPS, data$treatment))
  
  output$balance_df_mis = c(output$balance_df_mis, list(cb.mis))
  return (output)
  }

stopCluster(cl)

ate_HT = do.call(rbind, lapply(res, function(x) x$ate_HT))
colnames(ate_HT) = c('unadjusted', 'logistic', 'GLM_IS', 'NNet', 'NNet_IS', 'CBPS_exact')
write.csv(ate_HT, paste(location, 'ate_HT.csv', sep = ''))

ate_Hajek = do.call(rbind, lapply(res, function(x) x$ate_Hajek))
colnames(ate_Hajek) = c('unadjusted', 'logistic', 'GLM_IS', 'NNet', 'NNet_IS', 'CBPS_exact')
write.csv(ate_Hajek, paste(location, 'ate_Hajek.csv', sep = ''))

ate_DR = do.call(rbind, lapply(res, function(x) x$ate_DR))
write.csv(ate_DR, paste(location, 'ate_DR.csv', sep = ''))

bias_ps = do.call(rbind, lapply(res, function(x) x$bias_ps))
colnames(bias_ps) = c('logistic', 'GLM_IS', 'NNet', 'NNet_IS', 'CBPS_exact')
write.csv(bias_ps, paste(location, 'bias_ps.csv', sep = ''))

balance_df = do.call(c, lapply(res, function(x) x$balance_df))
saveRDS(balance_df, paste(location, 'balance_df_obs.rds', sep = ''))

balance_df_mis = do.call(c, lapply(res, function(x) x$balance_df_mis))
saveRDS(balance_df_mis, paste(location, 'balance_df_mis.rds', sep = ''))

balance_df_true = do.call(c, lapply(res, function(x) x$balance_df_true))
saveRDS(balance_df_true, paste(location, 'balance_df_true.rds', sep = ''))

best_tune_nn = do.call(c, lapply(res, function(x) x$best_tune_nn))
best_tune_glm = do.call(c, lapply(res, function(x) x$best_tune_glm))
best_tune = data.frame(nn = best_tune_nn,
                       glm = best_tune_glm)
write.csv(best_tune, paste(location, 'best_tune.csv', sep = ''))


time_diff = Sys.time() - start_time
print(time_diff)