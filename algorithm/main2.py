import os
os.chdir('/storage/home/yqs5519/work/TuneCalibratedLoss/algorithm')
import torch
import torch.optim as optim
import LossFunction
import TrainModel
import pandas as pd
import numpy as np
import sys


def process_data(i, location, base_model, ps_correct):
    if ps_correct == 'yes':
      file_path = location + "sim_data_true" + str(int(i)) + ".csv"
    if ps_correct == 'no':
      file_path = location + "sim_data_mis" + str(int(i)) + ".csv"
      
    column_names = ["treatment", "outcome", "x1", "x2", "x3", "x4"]
    df = pd.read_csv(file_path)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    trt_var = tensor[:, 1]
    y_train = tensor[:, 0]
    X_train = tensor[:, 2:]
    cont_vars = tensor[:, 2:]
    num_covariates = X_train.size()[1]

    train_dataset = torch.utils.data.TensorDataset(X_train, trt_var)
    Train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=X_train.size()[0])
    Test_dataloader = Train_dataloader

    torch.manual_seed(42)
    if base_model == 'NN_IS':
      Model_IS = LossFunction.Two_layer_LogisticRegression(input_dim=num_covariates, hidden_dim=num_covariates, output_dim=1)
    if base_model == 'GLM_IS':
      Model_IS = LossFunction.LogisticRegression(input_dim=num_covariates, output_dim=1)
    if base_model == 'NN_only':
      Model_IS = LossFunction.Two_layer_LogisticRegression(input_dim=num_covariates, hidden_dim=num_covariates, output_dim=1)
      tuning = 0
    
    Optimizer_IS = optim.SGD(Model_IS.parameters(), lr=0.00009, momentum=0)
    Loss_IS = LossFunction.likelihood_IS_loss
    
    ## 'NN_only'
    if base_model == 'NN_only':
      Results_IS = TrainModel.train_model(Model_IS, Loss_IS, Optimizer_IS, Train_dataloader, Test_dataloader, num_epochs=5000, lambda_para=0)
      pred_np = Results_IS.numpy()
      np.savetxt(location + 'NN_PS'+str(int(i))+'.csv', pred_np, delimiter=',')
    
    if base_model == 'GLM_IS' or base_model == 'NN_IS':
      results_list = []
      # tuning_range = np.round((np.arange(1, 6, 1)), 2) #original version
      tuning_range = np.round((np.arange(1, 3, 1)), 2) # just to check running time
      for lambda_tune in tuning_range:
          Results_IS = TrainModel.train_model(Model_IS, Loss_IS, Optimizer_IS, Train_dataloader, Test_dataloader, num_epochs=5000, lambda_para=lambda_tune)
          results_list.append((lambda_tune, Results_IS))
      
      pred = []
      # Loop over the results_list and save Results_IS as a dataframe
      for model in results_list:
          selected_model = model[1]
          pred_np = selected_model.numpy()
          df_pred = pd.DataFrame(pred_np)
          pred.append(df_pred)
      # Concatenate the dataframes into a single dataframe
      results_pred = pd.concat(pred, axis=1)
      results_pred.columns = ["Tune" + str(m) for m in tuning_range]
      
    if base_model == 'NN_IS':
      results_pred.to_csv(location + 'NN_IS_PS_results_pred'+str(int(i))+'.csv', index=False)
    if base_model == 'GLM_IS':
      results_pred.to_csv(location + 'GLM_IS_PS_results_pred'+str(int(i))+'.csv', index=False)

    


if __name__ == '__main__':
  i = sys.argv[1]
  location = sys.argv[2]
  base_model = sys.argv[3]
  ps_correct = sys.argv[4]
  
  process_data(i, location, base_model, ps_correct)

