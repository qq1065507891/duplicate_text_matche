import torch
from transformers import BertModel


class Question_Bert(torch.nn.Module):
    def __init__(self, config):
        super(Question_Bert, self).__init__()
        self.config = config
        self.model_name = config.model_name

        self.bert = BertModel.from_pretrained(config.bert_base_path)

        self.fc1 = torch.nn.Linear(self.bert.config.hidden_size, config.hidden_size)
        self.output = torch.nn.Linear(config.hidden_size, 2)
        self.dropout = torch.nn.Dropout(config.drop_out)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        :param inputs:{input_ids,  token_type_ids, attention_mask}
        :return:
        """
        input_ids, token_type_ids, attention_mask = inputs
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        first_hidden_states = last_hidden_states[:, 0, :]
        fc1 = self.fc1(first_hidden_states)
        drop_out = self.dropout(fc1)
        output = self.output(drop_out)
        # output = self.softmax(output)
        return output[:, 0]
