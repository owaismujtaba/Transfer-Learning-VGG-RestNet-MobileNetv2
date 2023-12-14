from src.dataloader import ArabicDatasetLoader
from src.models import V
from utils import load_processed_dataset, save_trained_model
from src.models import VGG19Backend, ResNet50Backend, MobileNetV2Backend
from src.trainer import train_model
import config

from utils import plot_train_history

if __name__=='__main__':
    
    X_train, y_train, X_val, y_val = load_processed_dataset()
    dataloader = ArabicDatasetLoader(X_train, y_train, X_val, y_val)
    
    train_loader = dataloader.train_dataset
    val_loader = dataloader.val_dataset
    
    
    if config.vgg16:
        model = VGG19Backend()
        model_name = 'VGG19'
        print(model.summary())
        
    elif config.restnet:
        model = ResNet50Backend()
        model_name = 'RestNet50'
        print(model.summary())
              
    else: 
        model = MobileNetV2Backend()
        model_name = 'MobileNetv2'
        print(model.summary())   
        
    history = train_model(
        model=model, 
        train_laoder=train_loader, 
        validation_loader=val_loader
    )
    
    plot_train_history(model, model_name)
    
    
    
    