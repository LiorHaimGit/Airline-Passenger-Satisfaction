# main.py

from src.data import make_dataset
from src.models.train_and_save_model import train_and_save_model
from src.models.ann_models import AnnModel
def main():
    # Load the data
    train_data, _ = make_dataset.load_data()

    # dataAnalysis(train_data)

    # Creating model
    input_size = len(train_data.columns) - 1
    model = create_model(input_size)

    # Train the model and save it
    train_and_save_model(train_data,model)

def create_model(input_size):
    # Create and return an instance of AnnModel with the specified parameters
    hidden_size = 32
    num_classes = 2
    return AnnModel(input_size, hidden_size, num_classes)

if __name__ == '__main__':
    main()
