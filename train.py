import torch
import models
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle




class Trainer:
    def __init__(self, is_cuda=True, get_batch_func=None, get_test_batch=None, checkpoint_path='./data', path_to_normalizer='./scaler.obj',global_loss=10000):
        self.is_cuda = is_cuda
        self.model = models.ResNetDepth(num_elements=199)
        if self.is_cuda:
            self.model = self.model.cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), 15e-7)
        self.loss = torch.nn.MSELoss()
        self.get_batch_func = get_batch_func
        self.get_test_batch = get_test_batch
        self.checkpoint_path = checkpoint_path
        self.normalizer = self.get_normalizer(path_to_normalizer)
        self.global_loss = global_loss

    def get_normalizer(self, path):
        file = open(path, 'rb')
        obj = pickle.load(file)
        file.close()
        return obj

    def save(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def resume(self):
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        except Exception as ex:
            print('no saved model: ', ex)

    def eval(self,X):
        with torch.no_grad():
            output = self.model(X)
            return self.inverse_transform(output)

    def train(self, epoches=1000, batch_size=10, train_data_count=1000, test_data_count=1000):
        self.resume()
        self.model.train()
        global_loss = self.global_loss
        for epoch in range(epoches):

            if epoch > 0 and self.get_test_batch:
                print("start testing")
                loss = 0
                for iteration in range(test_data_count // batch_size):
                    input, output = self.get_test_batch(iteration, batch_size)
                    output = self._transform_output(output)
                    if self.is_cuda:
                        input, output = input.cuda(), output.cuda()

                    with torch.no_grad():
                        prediction = self.model(input)
                        current_loss = self.loss(prediction, output).data
                        print('test loss', current_loss)
                        loss += current_loss


                    if global_loss < loss:
                        self.save()
                loss /= batch_size * iteration
            for iteration in range(train_data_count // batch_size):

                input, output = self.get_batch_func(batch_size)
                output = self._transform_output(output)
                if self.is_cuda:
                    input, output = input.cuda(), output.cuda()

                prediction = self.model(input)
                loss = self.loss(prediction, output)
                print(loss.item())

                loss.backward()
                self.optimizer.step()

                del loss

    def inverse_transform(self,y):
        return self.normalizer.inverse_transform(y)

    def _transform_output(self, y):
        return torch.from_numpy(self.normalizer.transform(y)).type(torch.FloatTensor)


