import random
import json
import torch
from chat_module import ChatNeuralNet
from Helper import NLP_helper
from voice import Voice
chat_voice=Voice()
helper=NLP_helper()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_module():
    global pattern_words, tags,intents
    with open('data.json', 'r') as json_data:
        intents = json.load(json_data)
    FILE = "chat_data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    pattern_words = data['pattern_words']
    tags = data['tags']
    model_state = data["model_state"]
    model = ChatNeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model
def get_response(model,sentence):    
    sentence = helper.sentence_tokenzation(sentence)
    X = helper.sentence_vector(sentence, pattern_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
               return random.choice(intent['responses']), tag
    else:
       return "I do not understand...", ' '
if __name__ == '__main__':
    model = load_module()
    chat_voice.speak('Hi I am liza your chat bot, how can i help you ', 140)
    while True:
        sentence = chat_voice.parscommand()
        
        response=get_response(model,sentence)
        tag=response[1]
        chat_voice.speak(response[0])
        if (tag=="goodbye"):
            break