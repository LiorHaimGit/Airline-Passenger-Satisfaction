def save_model(model, path):
    # Save the model
    torch.save(model.state_dict(), path)