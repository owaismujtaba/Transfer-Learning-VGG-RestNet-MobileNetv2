from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import config

def train_model(model, train_laoder, validation_loader):
    print('Training the model')
    import pdb
    pdb.set_trace()
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
   
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_laoder,
        epochs=config.epochs,
        callbacks=[early_stopping],
        validation_data=validation_loader
    )
    return history, model