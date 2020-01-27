import matplotlib.pyplot as plt

class GraphPlot():
    def __init__(self, model,save_model, x_val, y_val, z_val):
        self.model = model
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.save_model = save_model

    def plot(self):
        fig, ((ax1), (ax2)) = plt.subplots(1,2, figsize = (18,8))
        ax1.plot(self.z_val, self.x_val, color='green',linewidth=5)
        ax1.grid(True)
        ax1.set_xlabel('x',fontsize=32)
        ax1.set_ylabel('f(x)', fontsize=32)
        for label in (ax1.get_xticklabels()+ax1.get_yticklabels()):
            label.set_fontsize(22)
        if self.model == 'relu':
            ax1.set_title('ReLU\n', fontsize=28)
            ax1.set_ylim([-0.5, 1.5])
            ax1.set_xlim([-5,5])
        elif self.model == 'step':
            ax1.set_title('Binary Step\n', fontsize=28)
            ax1.set_ylim([-0.5, 1.5])
            ax1.set_xlim([-5,5])
        elif self.model == 'tanh':
            ax1.set_title('TanH\n', fontsize=28)
            ax1.set_ylim([-1.5, 1.5])
            ax1.set_xlim([-5,5])
        elif self.model == 'sigmoid':
            ax1.set_title('Sigmoid\n', fontsize=28)
            ax1.set_ylim([-0.5, 1.5])
            ax1.set_xlim([-5,5])
        elif self.model == 'linear':
            ax1.set_title('Linear\n', fontsize=28)
            ax1.set_ylim([0, 10])
            ax1.set_xlim([0,10])
        elif self.model == 'leaky':
            ax1.set_title('Leaky ReLU\n', fontsize=28)
            ax1.set_ylim([-2, 5])
            ax1.set_xlim([-5,5])

        ax2.plot(self.z_val,self.y_val, color='red', linewidth=5)
        ax2.grid(True)
        ax2.set_xlabel('x',fontsize=32)
        ax2.set_ylabel('f(x)', fontsize=32)
        for label in (ax2.get_xticklabels()+ax2.get_yticklabels()):
            label.set_fontsize(22)
        if self.model == 'relu':
            ax2.set_title('Derivative of ReLU\n', fontsize=28)
            ax2.set_ylim([-0.5, 1.5])
            ax2.set_xlim([-5,5])
        elif self.model == 'step':
            ax2.set_title('Derivative of Binary Step\n', fontsize=28)
            ax2.set_ylim([-0.5, 1.5])
            ax2.set_xlim([-5,5])
        elif self.model == 'tanh':
            ax2.set_title('Derivative of TanH\n', fontsize=28)
            ax2.set_ylim([-1.5, 1.5])
            ax2.set_xlim([-5,5])
        elif self.model == 'sigmoid':
            ax2.set_title('Derivative of Sigmoid\n', fontsize=28)
            ax2.set_ylim([-0.5, 1.5])
            ax2.set_xlim([-5,5])
        elif self.model == 'linear':
            ax2.set_title('Derivative of Linear\n', fontsize=28)
            ax2.set_ylim([0, 10])
            ax2.set_xlim([0,10])
        elif self.model == 'leaky':
            ax2.set_title('Derivative of Leaky ReLU\n', fontsize=28)
            ax2.set_ylim([-2, 5])
            ax2.set_xlim([-5,5])

        plt.subplots_adjust(bottom=0.15, wspace=0.5)
        if self.save_model == True:
            filename = 'plots/' + self.model + '.png'
            plt.savefig(filename, dpi=100)
        plt.show()
