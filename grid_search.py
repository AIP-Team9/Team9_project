import torch
from sklearn.model_selection import ParameterGrid
import torch.optim as optim
import eval_vis


def grid_search(model,train_data, train_label ,dataloader, param_grid , input_dim, output_dim , num_epochs = 10 , batch_size = 32) :
    results = []

    param_list list(ParameterGrid(param_grid))

    for params in param_list :
        model_1 = model(input_dim , param_list, output_dim)
        criterion = nn.MSELoss()
        optimizer = getattr(optim, params['optimizer'])(model_1.parameters(), lr = params['lr'])


        for epoch in range(num_epochs) :
            model.train()
            for inputs, labels in dataloader :
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad() :
            outputs = model_1(train_data)
            outputs = outputs.squeeze().numpy()
            labels = train_labels.numpy()
            result = eval_vis.evaluate_model(train_label , labels)


        results.append({'params':params , 'rmse' : result[0] , 'r2' : result[1]})

        best_result_rmse = min(results, key = lambda x: x['rmse'])
        best_result_r2 = min(results, key = lambda x: x['r2'])

        print("Best Parameters for RMSE : " , best_result_rmse['params'])
        print("Best RMSE : " , best_result_rmse['rmse'])

        print("Best Parameters for R2 : " , best_result_r2['params'])
        print("Best R2 : " , best_result_r2['r2'])


        return best_result_rmse , best_result_r2
