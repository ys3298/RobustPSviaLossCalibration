## helper function for model training
mean_out_trted = function(treatment, outcome, PS) {
  # estimate the mean outcome in the treated group by PS weighting method
  if (is.null(PS)) {
    y_mean = sum(treatment*outcome)/sum(treatment) 
  } else {y_mean = sum(treatment*outcome/PS)/sum(treatment/PS) }
  return(y_mean)
}

mean_out_ctl = function(treatment, outcome, PS) {
  # estimate the mean outcome in the treated group by PS weighting method
  if (is.null(PS)) {
    y_mean = sum((1-treatment)*outcome)/sum(1-treatment)
  } else {y_mean = sum((1-treatment)*outcome/(1-PS))/sum((1-treatment)/(1-PS)) }
  return(y_mean)
}

mean_ate_HT = function(treatment, outcome, PS) {
  # estimate the mean outcome in the treated group by PS weighting method
  if (is.null(PS)) {
    ate_mean = mean(treatment*outcome)-mean((1-treatment)*outcome)
  } else {ate_mean = mean(treatment*outcome/PS) -mean((1-treatment)*outcome/(1-PS)) }
  return(ate_mean)
}


mean_ate_Hajek = function(treatment, outcome, PS) {
  # estimate the mean outcome in the treated group by PS weighting method
  if (is.null(PS)) {
    ate_mean = sum(treatment*outcome)/sum(treatment) - sum((1-treatment)*outcome)/sum(1-treatment)
  } else {ate_mean = sum(treatment*outcome/PS)/sum(treatment/PS) - sum((1-treatment)*outcome/(1-PS))/sum((1-treatment)/(1-PS))}
  return(ate_mean)
}

mean_ate_DR = function(treatment, outcome, PS, m1, m0) {
  if (is.null(PS)) {
    ate_mean = mean(m1-m0 + (treatment*(outcome-m1) - (1-treatment)*(outcome-m0)))
  } else {ate_mean = mean(m1-m0 + (treatment*(outcome-m1)/PS - (1-treatment)*(outcome-m0)/(1-PS)))}
  
  return(ate_mean)
}
weight_X = function(X_orig, PS, treatment) {
  if (is.null(PS)) {
    weight = rep(1,nrow(X_orig))
  } else {
    df = data.frame(ps = PS, trt = treatment) %>% 
      mutate(w = ifelse(trt == 1, 1/ps, 1/(1-ps)))
    weight = df$w
  }
  X_weight = X_orig*weight
  
  return(X_weight)
}

weighted_mean_diff = function(X_orig, PS, treatment) {
  # w_X = weight_X(X_orig, PS, treatment)
  # w_X_treated = w_X[treatment == 1,]
  # w_X_control = w_X[treatment == 0,]
  # 
  # mean_X_treated = apply(w_X_treated,2,mean)
  # mean_X_control = apply(w_X_control,2,mean)
  # 
  # var_X_treated = apply(w_X_treated,2,var)
  # var_X_control = apply(w_X_control,2,var)
  # 
  # ASD = (mean_X_treated - mean_X_control) / sqrt(var_X_treated/nrow(w_X_treated) + var_X_control/nrow(w_X_control))
  # return(ASD)
  weight_subject = ifelse(treatment == 1, 1/PS, 1/(1-PS))
  group1 <- (treatment == 1)
  group0 <- (treatment == 0)
  
  asd = c()
  for (col in 1:dim(X_orig)[2]) {
    mean1 <- sum(X_orig[group1, col] * weight_subject[group1]) / sum(weight_subject[group1])
    mean0 <- sum(X_orig[group0, col] * weight_subject[group0]) / sum(weight_subject[group0])
    diff <- mean1 - mean0
    
    v1 <- sum(weight_subject[group1] * (X_orig[group1, col] - mean1) ^ 2) / (sum(weight_subject[group1]))
    v0 <- sum(weight_subject[group0] * (X_orig[group0, col] - mean0) ^ 2) / (sum(weight_subject[group0]))
    asd[col] <- (diff) / sqrt(v1 / sum(weight_subject[group1]) + v0 / sum(weight_subject[group0]))
  }
  return(asd)
}

pred_error = function(ps_pred, ps_true) {
  mse = mean((ps_pred - ps_true)^2)
  return(mse)
}

rmvbin_direct = function(n, margprob, sigma) {
  thresh = qnorm(margprob, sd = sqrt(diag(sigma)))
  Z = rmvnorm(n, mean = rep(0, length(margprob)), sigma = as.matrix(sigma))
  X = matrix(NA, ncol = ncol(Z), nrow = nrow(Z))
  for(i in 1:length(margprob)) {
    X[,i] = ifelse(Z[,i] <= thresh[i], 1, 0)
  }
  X_mis = cbind(X[,1]*X[,2],
                X[,1]*X[,1],
                X[,2]*X[,3],
                X[,2]*X[,4])
  return(list(X = X, X_mis = X_mis))
}

rmvbin_latent = function(n, margprob, sigma) {
  thresh = qnorm(margprob, sd = sqrt(diag(sigma)))
  Z = rmvnorm(n, mean = rep(0, length(margprob)), sigma = as.matrix(sigma))
  X = matrix(NA, ncol = ncol(Z), nrow = nrow(Z))
  for(i in 1:length(margprob)) {
    X[,i] = ifelse(Z[,i] <= thresh[i], 1, 0)
  }
  Z_star = cbind((Z[,1]*Z[,4]*Z[,3]),
                 Z[,2]*(Z[,1])^(-1),
                 (Z[,1]*Z[,3]+.6)^3,
                 (Z[,2]+Z[,4])*Z[,3])
  X_mis = matrix(NA, ncol = ncol(Z), nrow = nrow(Z))
  for(i in 1:length(margprob)) {
    X_mis[,i] = ifelse(Z_star[,i] <= thresh[i], 1, 0)
  }
  return(list(X = X, X_mis = X_mis))
}