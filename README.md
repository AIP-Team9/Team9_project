This is for AI Project team project.


Data Pipeline[data_process.py] : Data loading, Data preprocessing, Data visualization, Data save(Make new file after processing)
   
    - input : .csv file
    
    - output : processed .csv files(train/test/valid[optional])



Evaluation and Visualization[eval_vis.py] : Make evaluation score for each result of three models and visualize the result for comparing the effects of models
    
    - input : np.array which is the result of each models
    
    - Calculate RMSE, R-Squared
    
    - Visualizer these scores with graph



Three models : MLP[MLP.py] , LSTM[LSTM.py] , Transformer[Transformer.py]
    
    - Develop customized model for regression based on nn.Module
    
    - forward, train, eval method
    
    - eval output : np.array




Main[On colab notebook] :
        
    - Preprocess Data 
        
    - load train , test , valid[optional] dataset
        
    - Train three models with train() method
        
    - Evaluation and Visualization
        - Grid Search for hyperparameterization
        
