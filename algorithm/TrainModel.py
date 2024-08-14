import torch
import csv


def train_model(model, loss_function, optimizer, train_dataloader, test_dataloader, num_epochs, weight=None,
                lambda_para=1):
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(data)
            # print(outputs.max())
            loss = loss_function(outputs, targets, data, weight, lambda_para)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch + 1 >= num_epochs:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")
            
        loss_history.append(running_loss)

    # Save loss history as a CSV file
    # loss_csv_file = "/storage/work/yqs5519/NN_ImbalancePS/loss.csv"
    # with open(loss_csv_file, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Epoch", "Loss"])
    #     for epoch, loss in enumerate(loss_history):
    #         writer.writerow([epoch + 1, loss])

    predicted_probs = []
    with torch.no_grad():
        for data, targets in test_dataloader:
            outputs = model(data)
            predicted_prob = torch.sigmoid(outputs)
            predicted_probs.append(predicted_prob[:, 0])
        predicted_probs = torch.cat(predicted_probs, dim=0)
    return predicted_probs
