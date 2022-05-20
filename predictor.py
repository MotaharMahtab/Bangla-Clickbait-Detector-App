import torch
from transformers import AutoTokenizer
from config import device,num_hidden_layers_d,num_hidden_layers_g,out_dropout_rate
from model import GAN


class PythonPredictor:
    def __init__(self,
                  label_list=[0,1,2],):
        self.device = device
        self.label_list = label_list
        self.num_classes = len(self.label_list)
        self.labels = ['Non Clickbait','Clickbait']
        checkpoint = torch.load('clickbait_model.pt',map_location=torch.device(self.device))
        self.model = GAN()
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
    def predict(self, input_ids,input_mask_array):
      label,probability = self.model(input_ids.to(self.device),input_mask_array.to(self.device))
      return self.labels[label],probability.item()

# if __name__ == '__main__':
#   predictor = PythonPredictor()
#   print(predictor.predict('Viral News: অলৌকিক ঘটনা! একই সন্তানের দু-দুবার জন্ম দিলেন মা! নিজেই জানালেন আশ্চর্য কাহিনী'))