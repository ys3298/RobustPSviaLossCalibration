import torch
import torch.nn as nn
import torch.nn.functional as F

class Two_layer_LogisticRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Two_layer_LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
      
def absolute_standardized_difference(matrix, group_indicator, weight_covariate, weight_subject):
    """
    Calculate the absolute standardized difference of each column in a matrix and sum them together,
    using a group indicator vector and weight vector.

    Args:
    - matrix: tensor of shape (n, p), where n is the number of samples and p is the number of features
    - group_indicator: tensor of shape (n,), where each element is an integer indicating the group membership of the corresponding sample
    - weight: tensor of shape (p,), where each element is a weight for the corresponding feature

    Returns:
    - asd: scalar, weighted sum of elements where each element is the absolute standardized difference of the corresponding feature
    """
    group1 = (group_indicator == 1)
    group0 = (group_indicator == 0)
    n1 = group1.sum()
    n0 = group0.sum()

    asd = torch.zeros(matrix.shape[1], dtype=torch.float32)
    for col in range(matrix.shape[1]):
        mean1 = torch.sum(matrix[group1, col] * weight_subject[group1]) / torch.sum(weight_subject[group1])
        mean0 = torch.sum(matrix[group0, col] * weight_subject[group0]) / torch.sum(weight_subject[group0])
        diff = mean1 - mean0

        v1 = torch.sum(weight_subject[group1] * (matrix[group1, col] - mean1) ** 2) / (torch.sum(weight_subject[group1])-1)
        v0 = torch.sum(weight_subject[group0] * (matrix[group0, col] - mean0) ** 2) / (torch.sum(weight_subject[group0])-1)
        asd_temp = torch.abs(diff) / torch.sqrt(v1 / torch.sum(weight_subject[group1]) + v0 / torch.sum(weight_subject[group0]))
        asd[col] = asd_temp

    return torch.sum(asd*weight_covariate)


  
# Define the custom loss function
def likelihood_IS_loss(outputs, targets, X_data, weight=None, lambda_para = 1):
    # outputs is X'beta linear combination
    # X_data is the training covariates matrix
    # targets is the training label
    # weight is the weight for each covariate of data matrix, default is all 1
    # lambda_para is the tuning parameter that control the proportion of imbalance panelty

    if weight is None:
        weight = torch.ones(X_data.shape[1])

    trt_indicator = targets.view(-1)
    targets = targets.view(-1)
    outputs = outputs.view(-1)
    activation = torch.sigmoid(outputs).view(-1) # P(y=1|X,beta), n*1
    pi = torch.exp(outputs) / (1 + torch.exp(outputs)) # e^X'beta / 1 + e^X'beta, n*1
    w = trt_indicator / pi + (1 - trt_indicator) / (1 - pi) # n*1
    X_tilde = torch.mul(X_data.t(), w).t() # n*p

    # 1. ASD loss
    IS = absolute_standardized_difference(X_data, trt_indicator, weight, w)
        
    ## add with llh
    log_likelihood_loss = -torch.sum(targets * torch.log(activation) + (1 - targets) * torch.log(1 - activation))
    loss = log_likelihood_loss + lambda_para*IS
    
    return loss



def logistic_regression_likelihood_loss(outputs, targets, X_data, weight = None, lambda_para = 1):
    activation = torch.sigmoid(outputs).view(-1)
    log_likelihood_loss = -torch.sum(targets * torch.log(activation) + (1 - targets) * torch.log(1 - activation))
    return log_likelihood_loss


