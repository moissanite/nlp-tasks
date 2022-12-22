from torch import nn
from transformers import XLMRobertaModel, AutoConfig


class XlmRoberta(nn.Module):
    def __init__(self, model_name, n_labels):
        super(XlmRoberta, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        self.config_model = AutoConfig.from_pretrained(model_name)

        self.fc = nn.Linear(self.config_model.hidden_size, n_labels)
        self.softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ids, mask):
        outputs = self.roberta(input_ids=ids, attention_mask=mask)
        x = self.fc(outputs['pooler_output'])
        x = self.softmax(x)
        return x
