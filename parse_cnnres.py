"""
Parse and Plot results of Keras example "cifar10_cnn".
The Program also can be used for other results returned by Keras,
 but need to modify re pattern slightly.
"""
import re


class Result(object):
    def __init__(self, res_paths, epochs):
        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        self.res_paths = res_paths
        self.epochs = epochs

    def parse_train_loss(self, res_path):
        res = open(res_path, 'r').read()
        loss_pat = re.compile(r'(?<=50000/50000).*?(?=\n)')
        return [float(i.group().strip().split(' - ')[-1].split(': ')[-1]) for i in loss_pat.finditer(res)]
        # print loss_pat.findall(res)

    def parse_acc(self, res_path):
        res = open(res_path, 'r').read()
        acc_pat = re.compile(r'(?<=Test accuracy: ).*?(?=\n)')
        return float(acc_pat.findall(res)[0])

    def parse_test_loss(self, res_path):
        res = open(res_path, 'r').read()
        loss_pat = re.compile(r'(?<=Test score: ).*?(?=\n)')
        return float(loss_pat.findall(res)[0])

    def show_fig(self):
        import matplotlib.pyplot as plt
        plt.figure(1)

        def plot_res(epochs, loss, accuracy, name, color):
            x = range(0, epochs)
            plt.plot(x, loss, color=color, label='%s=%.2f%%' % (name, accuracy))

        for i in range(len(self.res_paths)):
            loss = self.parse_train_loss(self.res_paths[i])
            accuracy = self.parse_acc(self.res_paths[i])
            plot_res(self.epochs, loss, accuracy*100, name=self.res_paths[i], color=self.colors[i])

        plt.title('Loss in Epochs')
        plt.legend(loc='upper right')
        plt.show()

res = Result(res_paths=['adam_32', 'adam_64', 'adadelta_32', 'adagrad_32'], epochs=10)
res.show_fig()
