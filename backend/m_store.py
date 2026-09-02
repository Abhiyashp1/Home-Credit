from inference import load_data,load_model,load_features
ml_models={}

def load_models():
    ml_models["model"] = load_model()
    ml_models["selected_features"] = load_features()
    ml_models["df"] = load_data()

def get_models():
    return ml_models["model"], ml_models["selected_features"], ml_models["df"]
