# src/feature_selection.py
import pandas as pd
import xgboost as xgb
from typing import List

def get_top_n_genes_by_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_genes_to_select: int, random_state: int) -> List[str]:

    print(f"3. Selecting top {n_genes_to_select} genes using XGBoost...")
    

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,       
        max_depth=3,              
        learning_rate=0.01,        
        use_label_encoder=False,
        n_jobs=-1,               
        random_state=random_state
    )
    

    print("   - Training XGBoost model to get feature importances...")
    model.fit(X_train, y_train.squeeze())

    importances = model.feature_importances_
    
  
    feature_importances = pd.DataFrame({
        'gene': X_train.columns,
        'importance': importances
    })
    
  
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    
  
    top_n_genes = feature_importances.head(n_genes_to_select)['gene'].tolist()
    
    print(f"   - Selected {len(top_n_genes)} genes successfully.")
    
    return top_n_genes