def load_model(path):
    # Load the model
    model = TheModelClass()  # replace with your model class
    model.load_state_dict(torch.load(path))
    model.eval()

    return model