from src import dataset
from src.dataloader import ArabicDatasetLoader
from utils import load_processed_dataset
import config

if __name__=='__main__':
    
    X_train, y_train, X_val, y_val = load_processed_dataset()
    dataloader = ArabicDatasetLoader(X_train, y_train, X_val, y_val)
    
    train_loader = dataloader.train_dataset
    val_loader = dataloader.val_dataset
    
    
    if 
    
    
    
    
    