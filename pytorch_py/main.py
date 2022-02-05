from src.trainning import traninng
from src.predict import predict, batch_predict
from src.utils.config import config

if __name__ == "__main__":
    traninng(config)
    # q = 'What is the step by step guide to invest in share market in india?'
    # b = 'What is the step by step guide to invest in share market?'
    # q = ['What is the step by step guide to invest in share market in india?', 'What is the story of Kohinoor (Koh-i-Noor) Diamond?']
    # b = ['What is the step by step guide to invest in share market?', 'What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?']
    # predict(q, b)
    # batch_predict(q, b)
