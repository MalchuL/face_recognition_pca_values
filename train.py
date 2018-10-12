import torch
import models
import torch.optim as optim


class Trainer:
    def __init__(self, is_cuda=True, get_batch_func=None, get_test_batch=None, checkpoint_path='./data'):
        self.is_cuda = is_cuda
        self.model = models.ResNetDepth(num_elements=199)
        if self.is_cuda:
            self.model = self.model.cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), 1e-3)
        self.loss = torch.nn.MSELoss()
        self.get_batch_func = get_batch_func
        self.get_test_batch = get_test_batch
        self.checkpoint_path = checkpoint_path
        self.skip_test = None

    def save(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def resume(self):
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        except Exception as ex:
            print('no saved model: ', ex)

    def train(self, epoches=1000, batch_size=10, train_data_count=1000, test_data_count=1000):
        self.resume()
        self.model.train()
        global_loss = 100000
        for epoch in range(epoches):

            if self.get_test_batch:
                print("start testing")
                loss = 0
                for iteration in range(test_data_count // batch_size):
                    input, output = self.get_test_batch(iteration, batch_size)
                    if self.is_cuda:
                        input, output = input.cuda(), output.cuda()

                    with torch.no_grad():
                        prediction = self.model(input)
                        current_loss = self.loss(prediction, output).data
                        loss += current_loss

                    print('test loss', loss)
                    #testing
                    if self.skip_test is not None and self.skip_test>0:
                        print(iteration," current element of ", self.skip_test)
                        if iteration > self.skip_test:
                            break

                    if global_loss < loss:
                        self.save()
                loss /= batch_size * iteration
            for iteration in range(train_data_count // batch_size):

                input, output = self.get_batch_func(batch_size)
                if self.is_cuda:
                    input, output = input.cuda(), output.cuda()

                prediction = self.model(input)
                loss = self.loss(prediction, output)
                print(loss.data)

                loss.backward()
                self.optimizer.step()

                del loss

    def eval(self):
        self.resume()
