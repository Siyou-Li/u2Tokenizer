from transformers import Phi3Config


class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"