from src.utils.process import process_pre_text
from src.model.InitModel import load_model
from src.utils.config import config

config = config


def predict(q1, q2):
    token_ids, segment_ids = process_pre_text(q1, q2)
    model = load_model(config)
    pre = model.predict((token_ids, segment_ids))
    return pre
