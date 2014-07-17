from SVM.DataHandler import DataHandler
import matplotlib.pyplot as plt


def scatterplot(data, data_name, c=None):
    """Makes a scatterplot matrix:
     Inputs:
         data - a list of data [dataX, dataY,dataZ,...]; 
                  all elements must have same length 
         data_name - a list of descriptions of the data;
                  len(data) should be equal to len(data_name)
    Output:
         fig - matplotlib.figure.Figure Object
    """
    N = len(data) 
    fig = plt.figure(figsize=(15, 10)) 
    for i in range(N):
        for j in range(N): 
            ax = fig.add_subplot(N,N,i*N+j+1) 
            if j == 0: 
                ax.set_ylabel(data_name[i],size='12')
            if i == 0: 
                ax.set_title(data_name[j],size='12')
            if i == j: 
                ax.hist(data[i], 10) 
            else: 
                if c is not None:
                    ax.scatter(data[j], data[i], c=c, alpha=0.5, cmap='rainbow') 
                else:
                    ax.scatter(data[j], data[i],  alpha=0.5, cmap='rainbow') 
    return fig


if __name__ == "__main__":
    data_handler = DataHandler()
    all_data, all_targets = data_handler.get_training_data()
    headers = data_handler.get_headers()

    fig = scatterplot(all_data[:4], headers[:4])
    fig.savefig('reports/pairFeatures.png', dpi=120)
    plt.show()

    # import numpy as np
    # import numpy.random as npr
    # X = npr.randn(100)
    # Y = 1.2 * X + npr.normal(0.0, 0.1, 100)
    # Z = - Y ** 2 + X + 0.05 * npr.random(100)
    # W = X + Y - Z + npr.normal(0.0, 2.0, 100)
    # data = [X, Y, Z, W]
    #
    # data_name = ['Data X', 'Data Y', 'Data Z', 'Data W']
    #
    # fig = scatterplot(data, data_name)
    # fig.savefig('reports/pairFeatures.png', dpi=120)
    # plt.show()