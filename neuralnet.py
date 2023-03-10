import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=True,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)

def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape): #shape: [rows, cols], shape include bias col
    np.random.seed(np.prod(shape))

    ini = np.random.uniform(-0.1,0.1,shape)
    ini[:, 0] = 0 # bias col
    return ini

def zero_init(shape):
    """ shape: [rows, cols]
    """
    return np.zeros(shape)

def sigmoid(x):
    #applying the sigmoid function element-wise to the input.
    e = np.exp(x)
    return e / (1 + e)

def softmax(b):
    #applying the softmax function element-wise to the input
    e = np.exp(b)
    return e / e.sum() #constant denominator

class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        #:param weight_init_fn: weight initialization function
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        self.al = weight_init_fn([hidden_size, input_size]) #D* M+1 -> n_input=M+1
        self.be = weight_init_fn([output_size, hidden_size+1]) # 10 * D + 1

        # forward intermid ans
        self.a = np.zeros(hidden_size)
        self.z = np.zeros(hidden_size) # with bias
        self.b = np.zeros(output_size) # K
        # self.yhat = np.zeros(output_size) --> if saves, validation cross_en will modify

        #  save backward
        self.gyhat = np.zeros(output_size)
        self.gb = np.zeros(output_size)
        self.gbe = np.zeros([output_size, hidden_size+1])# with bias
        self.gzstar = np.zeros(hidden_size)
        self.ga = np.zeros(hidden_size)
        self.gal = np.zeros([hidden_size, input_size+1])

        # initialize parameters for adagrad
        self.epsilon = 1e-5
        self.grad_sum_al = np.zeros([hidden_size, input_size]) # input size already within bias
        self.grad_sum_be = np.zeros([output_size, hidden_size+1])

def print_para(nn):
    logging.debug(f"shape of alpha: {nn.al.shape}")
    logging.debug(nn.al)
    logging.debug(f"shape of beta: {nn.be.shape}")
    logging.debug(nn.be)

def forward(X, nn): # X in M+1, nn - object
    #return: output probability
    # save each node to nn
    nn.a = np.matmul(nn.al, X) # D * 1
    z_star = sigmoid(nn.a) # D * 1
    nn.z = np.ones(nn.n_hidden + 1) # fold bias
    nn.z[1:] = z_star # D+1 * 1
    nn.b = np.matmul(nn.be, nn.z) # K * 1
    yhat = softmax(nn.b) # K * 1
    return yhat

def backward(X, y, yhat, nn):
    y_v = np.zeros(nn.n_output)
    y_v[y] = 1
    nn.gyhat = -1 * y_v / yhat
    nn.gb = yhat - y_v # K * 1 -vec
    nn.gbe = np.matmul(nn.gb.reshape(-1,1), nn.z.reshape(1,-1))
    be_star = nn.be[:,1:] # remove first col
    nn.gzstar = np.matmul(be_star.T, nn.gb) # D * 1 - vec
    z_star = nn.z[1:]
    nn.ga = z_star * (1- z_star) * nn.gzstar # D * 1 - vec
    nn.gal = np.matmul(nn.ga.reshape(-1,1), X.reshape(1,-1))
    # return nn.gal, nn.gbe
    return

def test(X, y, nn):
    """
    labels: predicted labels
    error_rate: prediction error rate
    """
    err = 0
    lbs = np.zeros(y.shape[0], dtype = int)
    for i in range(y.shape[0]):
        y_hat = forward(X[i], nn) # vector K * 1
        lbs[i] = np.argmax(y_hat)
        if lbs[i] != y[i]: err+=1
    return lbs, err / y.shape[0]


def train(X_tr, y_tr, nn, X_va, y_va): # X_tr - original data, with all rows
    # pass validation for calculate jfunc
    tr_j = []
    va_j = []
    for e in range(nn.n_epoch):
        Xs, ys = shuffle(X_tr, y_tr, e) # Xs_cur, ys_cur
        for i in range(Xs.shape[0]):
            X_cur, y_cur = Xs[i], ys[i]
            y_hat = forward(X_cur, nn)
            backward(X_cur, y_cur, y_hat, nn)
            nn.grad_sum_al += nn.gal * nn.gal
            nn.grad_sum_be += nn.gbe * nn.gbe
            nn.al = nn.al - nn.lr / np.sqrt(nn.grad_sum_al + nn.epsilon) * nn.gal
            nn.be = nn.be - nn.lr / np.sqrt(nn.grad_sum_be + nn.epsilon) * nn.gbe
            print_para(nn)
        jt, jv = 0, 0
        #print_shape(y_tr, y_va)
        for i in range(X_tr.shape[0]): 
            y_hat = forward(X_tr[i], nn)
            yi_tr = np.zeros(y_hat.shape[0])
            yi_tr[y_tr[i]] = 1 # y[i] the current label for train
            jt += (yi_tr * np.log(y_hat)).sum()
        for i in range(X_va.shape[0]):
            y_hat = forward(X_va[i], nn) 
            yi_va = np.zeros(y_hat.shape[0])
            yi_va[y_va[i]] = 1 # y[i] the current label for valid
            jv += (yi_va * np.log(y_hat)).sum()
        tr_j.append(-1*jt/X_tr.shape[0])
        va_j.append(-1*jv/X_va.shape[0])
    return tr_j, va_j
    #return nn.al, nn.be

if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr = args2data(args)

    # Build model
    if init_flag==1:
        my_nn = NN(lr, n_epochs, random_init, X_tr.shape[1], n_hid, 10)
    else:
        my_nn = NN(lr, n_epochs, zero_init, X_tr.shape[1], n_hid, 10)

    # train model
    train_js, valid_js = train(X_tr, y_tr, my_nn, X_te, y_te)
    m = open(out_metrics, 'w')
    for i in range(n_epochs):
        m.write(f"epoch={i+1} crossentropy(train): {train_js[i]}\n")
        m.write(f"epoch={i+1} crossentropy(validation): {valid_js[i]}\n")

    # test model and get predicted labels and errors
    tr_lbs, tr_err = test(X_tr, y_tr, my_nn)
    va_lbs, va_err = test(X_te, y_te, my_nn)

    # write predicted label and error into file
    np.savetxt(out_tr, tr_lbs, fmt = "%i", newline = "\n")
    np.savetxt(out_te, va_lbs, fmt = "%i", newline = "\n")
    m.write("error(train): {:.3f}\n".format(tr_err))
    m.write("error(validation): {:.3f}\n".format(va_err))
