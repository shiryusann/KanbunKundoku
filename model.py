import torch
from typing import List, Dict, Tuple
from torch import nn

class BaseKanbunModel(nn.Module):
    def __init__(self, bert:nn.Module, loss_function, bert_dimension:int, classes:List[int], dropout:float):
        super().__init__()
        self.bert_model = bert
        self.loss_fn = loss_function
        self.dropout = nn.Dropout(p = dropout)
        self.classifier_okurigana = nn.Linear(bert_dimension, classes[0])
        self.classifier_particle = nn.Linear(bert_dimension, classes[1])
        self.classifier_position = nn.Linear(bert_dimension, classes[2])

    def forward(self, sentences:Dict[str, torch.tensor], labels:Dict[str, torch.tensor]):
        x = self.bert_model(**sentences).last_hidden_state
        x = self.dropout(x)

        logits_okurigana = self.classifier_okurigana(x).transpose(1, 2)
        logits_particle = self.classifier_particle(x).transpose(1, 2)
        logits_position = self.classifier_position(x).transpose(1, 2)

        loss = 0
        if labels is not None:
            loss += self.loss_fn(logits_okurigana, labels["okurigana"])
            loss += self.loss_fn(logits_particle, labels["particle"])
            loss += self.loss_fn(logits_position, labels["position"])
        
        return loss, logits_okurigana, logits_particle, logits_position

class OneAuxiliaryTaskModel(nn.Module):
    def __init__(self, bert:nn.Module, loss_function, bert_dimension:int, classes:List[int], auxiliary_names:List[str], weights:Tuple[float, float], dropout:float):
        super().__init__()
        self.bert_model = bert
        self.loss_fn = loss_function
        self.auxiliary_names = auxiliary_names
        self.weights = weights
        self.dropout = nn.Dropout(p = dropout)
        self.classifier_task1 = nn.Linear(bert_dimension, classes[3])
        self.classifier_okurigana = nn.Linear(bert_dimension + classes[3], classes[0])
        self.classifier_particle = nn.Linear(bert_dimension + classes[3], classes[1])
        self.classifier_position = nn.Linear(bert_dimension + classes[3], classes[2])

    def forward(self, sentences:Dict[str, torch.tensor], labels:Dict[str, torch.tensor]):
        x = self.bert_model(**sentences).last_hidden_state
        x = self.dropout(x)

        logits_task1 = self.classifier_task1(x)
        input_after_task1 = torch.cat((x, logits_task1), -1)
        input_after_task1 = self.dropout(input_after_task1)

        logits_task1 = logits_task1.transpose(1, 2)
        
        logits_okurigana = self.classifier_okurigana(input_after_task1).transpose(1, 2)
        logits_particle = self.classifier_particle(input_after_task1).transpose(1, 2)
        logits_position = self.classifier_position(input_after_task1).transpose(1, 2)

        loss = 0
        if labels is not None:
            main_loss = 0
            main_loss += self.loss_fn(logits_okurigana, labels["okurigana"])
            main_loss += self.loss_fn(logits_particle, labels["particle"])
            main_loss += self.loss_fn(logits_position, labels["position"])

            auxiliary_loss = 0
            auxiliary_loss += self.loss_fn(logits_task1, labels[self.auxiliary_names[0]])

            loss = main_loss * self.weights[0] + auxiliary_loss * self.weights[1]
        
        return loss, logits_okurigana, logits_particle, logits_position

class TwoAuxiliaryTaskModel(nn.Module):
    def __init__(self, bert:nn.Module, loss_function, bert_dimension:int, classes:List[int], auxiliary_names:List[str], weights:Tuple[float, float], dropout:float):
        super().__init__()
        self.bert_model = bert
        self.loss_fn = loss_function
        self.auxiliary_names = auxiliary_names
        self.weights = weights
        self.dropout = nn.Dropout(p = dropout)
        self.classifier_task1 = nn.Linear(bert_dimension, classes[3])
        self.classifier_task2 = nn.Linear(bert_dimension + classes[3], classes[4])
        self.classifier_okurigana = nn.Linear(bert_dimension + classes[3] + classes[4], classes[0])
        self.classifier_particle = nn.Linear(bert_dimension + classes[3] + classes[4], classes[1])
        self.classifier_position = nn.Linear(bert_dimension + classes[3] + classes[4], classes[2])

    def forward(self, sentences:Dict[str, torch.tensor], labels:Dict[str, torch.tensor]):
        x = self.bert_model(**sentences).last_hidden_state
        x = self.dropout(x)

        logits_task1 = self.classifier_task1(x)
        input_after_task1 = torch.cat((x, logits_task1), -1)
        input_after_task1 = self.dropout(input_after_task1)

        logits_task2 = self.classifier_task2(input_after_task1)
        input_after_task2 = torch.cat((x, logits_task1, logits_task2), -1)
        input_after_task2 = self.dropout(input_after_task2)

        logits_task1 = logits_task1.transpose(1, 2)
        logits_task2 = logits_task2.transpose(1, 2)
        
        logits_okurigana = self.classifier_okurigana(input_after_task2).transpose(1, 2)
        logits_particle = self.classifier_particle(input_after_task2).transpose(1, 2)
        logits_position = self.classifier_position(input_after_task2).transpose(1, 2)

        loss = 0
        if labels is not None:
            main_loss = 0
            main_loss += self.loss_fn(logits_okurigana, labels["okurigana"])
            main_loss += self.loss_fn(logits_particle, labels["particle"])
            main_loss += self.loss_fn(logits_position, labels["position"])

            auxiliary_loss = 0
            auxiliary_loss += self.loss_fn(logits_task1, labels[self.auxiliary_names[0]])
            auxiliary_loss += self.loss_fn(logits_task2, labels[self.auxiliary_names[1]])

            loss = main_loss * self.weights[0] + auxiliary_loss * self.weights[1]
        
        return loss, logits_okurigana, logits_particle, logits_position

class ThreeAuxiliaryTaskModel(nn.Module):
    def __init__(self, bert:nn.Module, loss_function, bert_dimension:int, classes:List[int], auxiliary_names:List[str], weights:Tuple[float, float], dropout:float):
        super().__init__()
        self.bert_model = bert
        self.loss_fn = loss_function
        self.auxiliary_names = auxiliary_names
        self.weights = weights
        self.dropout = nn.Dropout(p = dropout)
        self.classifier_task1 = nn.Linear(bert_dimension, classes[3])
        self.classifier_task2 = nn.Linear(bert_dimension + classes[3], classes[4])
        self.classifier_task3 = nn.Linear(bert_dimension + classes[3] + classes[4], classes[5])
        self.classifier_okurigana = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5], classes[0])
        self.classifier_particle = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5], classes[1])
        self.classifier_position = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5], classes[2])

    def forward(self, sentences:Dict[str, torch.tensor], labels:Dict[str, torch.tensor]):
        x = self.bert_model(**sentences).last_hidden_state
        x = self.dropout(x)

        logits_task1 = self.classifier_task1(x)
        input_after_task1 = torch.cat((x, logits_task1), -1)
        input_after_task1 = self.dropout(input_after_task1)

        logits_task2 = self.classifier_task2(input_after_task1)
        input_after_task2 = torch.cat((x, logits_task1, logits_task2), -1)
        input_after_task2 = self.dropout(input_after_task2)

        logits_task3 = self.classifier_task3(input_after_task2)
        input_after_task3 = torch.cat((x, logits_task1, logits_task2, logits_task3), -1)
        input_after_task3 = self.dropout(input_after_task3)

        logits_task1 = logits_task1.transpose(1, 2)
        logits_task2 = logits_task2.transpose(1, 2)
        logits_task3 = logits_task3.transpose(1, 2)
        
        logits_okurigana = self.classifier_okurigana(input_after_task3).transpose(1, 2)
        logits_particle = self.classifier_particle(input_after_task3).transpose(1, 2)
        logits_position = self.classifier_position(input_after_task3).transpose(1, 2)

        loss = 0
        if labels is not None:
            main_loss = 0
            main_loss += self.loss_fn(logits_okurigana, labels["okurigana"])
            main_loss += self.loss_fn(logits_particle, labels["particle"])
            main_loss += self.loss_fn(logits_position, labels["position"])

            auxiliary_loss = 0
            auxiliary_loss += self.loss_fn(logits_task1, labels[self.auxiliary_names[0]])
            auxiliary_loss += self.loss_fn(logits_task2, labels[self.auxiliary_names[1]])
            auxiliary_loss += self.loss_fn(logits_task3, labels[self.auxiliary_names[2]])

            loss = main_loss * self.weights[0] + auxiliary_loss * self.weights[1]
        
        return loss, logits_okurigana, logits_particle, logits_position

class FourAuxiliaryTaskModel(nn.Module):
    def __init__(self, bert:nn.Module, loss_function, bert_dimension:int, classes:List[int], auxiliary_names:List[str], weights:Tuple[float, float], dropout:float):
        super().__init__()
        self.bert_model = bert
        self.loss_fn = loss_function
        self.auxiliary_names = auxiliary_names
        self.weights = weights
        self.dropout = nn.Dropout(p = dropout)
        self.classifier_task1 = nn.Linear(bert_dimension, classes[3])
        self.classifier_task2 = nn.Linear(bert_dimension + classes[3], classes[4])
        self.classifier_task3 = nn.Linear(bert_dimension + classes[3] + classes[4], classes[5])
        self.classifier_task4 = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5], classes[6])
        self.classifier_okurigana = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5] + classes[6], classes[0])
        self.classifier_particle = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5] + classes[6], classes[1])
        self.classifier_position = nn.Linear(bert_dimension + classes[3] + classes[4] + classes[5] + classes[6], classes[2])

    def forward(self, sentences:Dict[str, torch.tensor], labels:Dict[str, torch.tensor]):
        x = self.bert_model(**sentences).last_hidden_state
        x = self.dropout(x)

        logits_task1 = self.classifier_task1(x)
        input_after_task1 = torch.cat((x, logits_task1), -1)
        input_after_task1 = self.dropout(input_after_task1)

        logits_task2 = self.classifier_task2(input_after_task1)
        input_after_task2 = torch.cat((x, logits_task1, logits_task2), -1)
        input_after_task2 = self.dropout(input_after_task2)

        logits_task3 = self.classifier_task3(input_after_task2)
        input_after_task3 = torch.cat((x, logits_task1, logits_task2, logits_task3), -1)
        input_after_task3 = self.dropout(input_after_task3)

        logits_task4 = self.classifier_task4(input_after_task3)
        input_after_task4 = torch.cat((x, logits_task1, logits_task2, logits_task3, logits_task4), -1)
        input_after_task4 = self.dropout(input_after_task4)

        logits_task1 = logits_task1.transpose(1, 2)
        logits_task2 = logits_task2.transpose(1, 2)
        logits_task3 = logits_task3.transpose(1, 2)
        logits_task4 = logits_task4.transpose(1, 2)
        
        logits_okurigana = self.classifier_okurigana(input_after_task4).transpose(1, 2)
        logits_particle = self.classifier_particle(input_after_task4).transpose(1, 2)
        logits_position = self.classifier_position(input_after_task4).transpose(1, 2)

        loss = 0
        if labels is not None:
            main_loss = 0
            main_loss += self.loss_fn(logits_okurigana, labels["okurigana"])
            main_loss += self.loss_fn(logits_particle, labels["particle"])
            main_loss += self.loss_fn(logits_position, labels["position"])

            auxiliary_loss = 0
            auxiliary_loss += self.loss_fn(logits_task1, labels[self.auxiliary_names[0]])
            auxiliary_loss += self.loss_fn(logits_task2, labels[self.auxiliary_names[1]])
            auxiliary_loss += self.loss_fn(logits_task3, labels[self.auxiliary_names[2]])
            auxiliary_loss += self.loss_fn(logits_task4, labels[self.auxiliary_names[3]])

            loss = main_loss * self.weights[0] + auxiliary_loss * self.weights[1]
        
        return loss, logits_okurigana, logits_particle, logits_position