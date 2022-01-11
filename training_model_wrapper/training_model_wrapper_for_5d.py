from training_model_wrapper.training_model_wrapper import TrainingModelWrapper
from torch.autograd import Variable
import torch

class TrainingModelWrapperFor5d(TrainingModelWrapper):

    def calculate_train_loss(self):
        training_loss = 0
        for batch_idx, (inputs, *labels, label_weight, mask) in enumerate(self.training_generator):
            inputs = Variable(inputs.to(self.device))
            label_weight = Variable(label_weight.to(self.device))
            mask = Variable(mask.to(self.device))
            labels = list(labels)
            new_labels_list = []
            for idx, label in enumerate(labels):
                # label[0] is normal label
                # label[x] with x > 0 is ordinal label
                if idx == 0:
                    new_labels_list.append(Variable(label.to(self.device)))
                elif idx > 0:
                    label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                    new_label = Variable(label.to(self.device))
                    new_labels_list.append(new_label)
            predicts = self.network_architecture(inputs)
            loss = 0
            if "UncertaintyLoss" in str(type(self.lost_list[0])) or "TestLoss" in str(type(self.lost_list[0])):
                loss += self.lost_list[0](predicts, new_labels_list, label_weight, mask)
            elif len(self.lost_list) == 1:
                loss += self.lost_list[0](predicts, new_labels_list[0], label_weight, mask)
            else:
                for idx, (criteria, criteria_weight) in enumerate(zip(self.lost_list, self.loss_weights)):
                    predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                    if idx > 0:
                        loss = loss.to(criteria.device)
                    loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
            self.optimizer.zero_grad()
            loss.to(self.device)
            loss.backward()
            self.optimizer.step()
            training_loss += float(loss.item())
        return training_loss

    def calculate_validate_loss(self):
        validation_loss = 0
        self.network_architecture.eval()
        with torch.no_grad():
            for index, (inputs, labels, label_weight, mask) in enumerate(self.validation_generator):
                inputs = Variable(inputs.to(self.device))
                labels = list(labels)
                new_labels_list = []
                for idx, label in enumerate(labels):
                    # label[0] is normal label
                    # label[x] with x > 0 is ordinal label
                    if idx == 0:
                        new_labels_list.append(Variable(label.to(self.device)))
                    elif idx > 0:
                        label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                        new_label = Variable(label.to(self.device))
                        new_labels_list.append(new_label)
                label_weight = Variable(label_weight.to(self.device))
                mask = Variable(mask.to(self.device))
                predicts = self.network_architecture(inputs)
                loss = 0
                if "UncertaintyLoss" in str(type(self.val_loss_list[0])) or "TestLoss" in str(type(self.val_loss_list[0])):
                    loss += self.val_loss_list[0](predicts, new_labels_list, label_weight, mask)
                elif len(self.val_loss_list) == 1:
                    if type(predicts) is tuple:
                        loss += self.val_loss_list[0](predicts[0], new_labels_list[0], label_weight, mask)
                    else:
                        loss += self.val_loss_list[0](predicts, new_labels_list[0], label_weight, mask)
                else:
                    for idx, (criteria, criteria_weight) in enumerate(zip(self.val_loss_list, self.val_loss_weights)):
                        predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                        loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
                validation_loss += loss
        if "UncertaintyLoss" in str(type(self.val_loss_list[0])):
            for idx, p in enumerate(self.val_loss_list[0].params):
                print(f"{idx}: {p}")
        return validation_loss
