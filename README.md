# my-Back-Propagation
README

API

Introduction

This code provides a simple but efficient approach for automatic back-propagation for any given computational graph of any topological structure. Compared with prevailing deep learning frameworks like Tensor Flow or PyTorch, it is short enough, easy enough, and flexible enough for beginners to get familiar with the principles of back-propagation and different network structures, including FC, Conv, ReLU, Softmax, Cross-Entropy Loss, Skip Connections for ResNet. It is implemented from scratch with python without importing any other libraries apart from numpy and queue.  

Low level API: automatic back-propagation for scalar computational graph

You may use the low level API via node and graph class. For example, if you would like to back-prop the following computational graph:

y_o = (\sin(x_1+1)+\cos(2x_2))\tan(\log(x_3))+(\sin(x_2+1)+\cos(2x_1))\exp(1+\sin(x_3))

you may just define the forward path of the computational graph with the node class:

    # construct forward computational graph with APIs in node and graph classes
    x1 = node('input', None)
    x2 = node('input', None)
    x3 = node('input', None)
    c1 = node('const', None)
    c2 = node('const', None)
    fac1 = node('+', node('sin', node('+', x1, c1)), 
                        node('cos', node('*', c2, x2)))
    fac2 = node('tan', node('log', x3))
    fac3 = node('+', node('sin', node('+', x2, c1)), 
                        node('cos', node('*', c2, x1)))
    fac4 = node('exp', node('+', c1, node('sin', x3)))
    o = node('+', node('*', fac1, fac2), node('*', fac3, fac4), is_output = True)

Then, you may run the graph with the graph class:

    # compute
    G = graph([x1,x2,x3], [c1,c2,c3], [o])
    x_init = np.random.random(size = 3)
    const_init = [1.0, 2.0, 1.0]
    G.feed(x_init, const_init)
    G.compute_forward()
    G.back_prop()

You may get the forward or backward value of any node effortlessly:

    yo = o.forward
    gradient = [x1.backward, x2.backward, x3.backward]

Higher level API: automatic back-propagation for Neural Networks

You may use the higher level API via nn class, which is built upon the previous classes. For example, if you would like to back-prop a simple ResNet inluding the following structure (graph is copied from the original ResNet paper (Kaiming He, Deep Residual Learning for Image Recognition, CVPR 2016)):



you may just define the forward path of the computational graph with the nn class:

    # hyper-parameters
    size = 16   # the size of input picture will be of size by size
    classes = 10    # how many classes there are in the classification task
    FC_nodes = 32   # how many hidden nodes there are in the Fully-Connected layer
    Conv_size = 9   # size of convolution operator
    
    # construct ResNet forward computational graph with higher APIs from nn class
    ResNet = nn()
    image = ResNet.new_const_nodes_list(size * size)
    label = ResNet.new_const_nodes_list(classes)
    zero = ResNet.new_const_nodes_list(1)
    
    W1 = ResNet.new_input_nodes_list(Conv_size)
    b1 = ResNet.new_input_nodes_list(1)
    Conv1 = ResNet.Conv(size, size, image, W1, b1, zero)[0]
    ReLU1 = ResNet.ReLU(size * size, Conv1)
    
    W2 = ResNet.new_input_nodes_list(Conv_size)
    b2 = ResNet.new_input_nodes_list(1)
    Conv2 = ResNet.Conv(size, size, ReLU1, W2, b2, zero)[0]
    
    Sum2 = ResNet.vector_add(size * size, image, Conv2) # skip connections
    ReLU2 = ResNet.ReLU(size * size, Sum2)
    
    W3 = ResNet.new_input_nodes_list(size * size * FC_nodes)
    b3 = ResNet.new_input_nodes_list(FC_nodes)
    FC3 = ResNet.FC(size * size, FC_nodes, ReLU2, W3, b3)[0]
    ReLU3 = ResNet.ReLU(FC_nodes, FC3)
    
    W4 = ResNet.new_input_nodes_list(FC_nodes * classes)
    b4 = ResNet.new_input_nodes_list(classes)
    FC4 = ResNet.FC(FC_nodes, classes, ReLU3, W4, b4)[0]
    Softmax4 = ResNet.Softmax(classes, FC4)
    
    Loss = ResNet.Cross_Entropy_Loss(classes, Softmax4, label)
    
    Loss.set_as_output()
    
    Weights = W1 + b1 + W2 + b2 + W3 + b3 + W4 + b4
    params = len(Weights)
    Graph = graph(Weights, image + label + zero, [Loss])

Then, you may run the graph with the graph class:

    # initialize
    init_weights = np.random.random(size = Conv_size + 1 + Conv_size + 1
                                    + size * size * FC_nodes + FC_nodes
                                    + FC_nodes * classes + classes) - 0.5
    init_image = np.random.random(size = size * size)
    init_label = [0,1,0,0,0,0,0,0,0,0]
    init_zero = [0]
    
    Graph.feed(list(init_weights), list(init_image) + init_label + init_zero)
    
    # compute
    Graph.compute_forward()
    Graph.back_prop()

Code Structure

Now, we provide a brief overview of the code structure, in case you may want to extend the code.

node class

    # computational node in DAG, similar to placeholder of Tensor Flow
    class node:
    
        # constructor
        def __init__(self, operator, left_child, right_child = None, is_output = False):
            # ...
            self.forward = 0    # forward value
            self.backward = 0   # output gradient w.r.t. this node
            self.parents = []   # pointers to parents nodes
            self.left_child = None  # pointers to left child nodes
            self.right_child = None # pointers to right child nodes
            self.operator = ''  # operator type, e.g., input, const, +, *, sin, ...
                                # operator is either input, const, unary or binary
            self.cnt_parents = 0    # how many parents it has
            self.cnt_ref = 0    # reference count, how many gradients w.r.t. this
                                # node is added to self.backward
            self.init = False   # if the node is fed with a value
            self.bp_ed = False  # if back-prop from this node to its children nodes
                                # has completed
            self.is_output = is_output  # if the node is the output node
            self.visited = 0    # for use in clear function when traversing
            # ...
    
        # ...
        
        # feed value, initiate
        def feed(self, value):
            # ...
    
        # compute forwardly: compute the value of this node forwarding
        # this function will return the node's parents nodes who is ready to compute forwardly
        def compute_forward(self):
            # ...
    
        # computing gradient w.r.t the child of this node, adding it to that
        # this function will return the node's children nodes who is ready to back-prop
        def back_prop(self):
            # ...

graph class

    # computational graph presented as DAG
    class graph:
    
        # constructor
        def __init__(self, input_nodes, const_nodes, output_nodes):
            self.input_nodes = input_nodes
            self.const_nodes = const_nodes
            self.output_nodes = output_nodes
            self.visited_ind = 1    # the nodes in the graph is visited during a traversing
                                    # if its visited bit is the same as self.visited indicator
    
        # feed value, initiate
        def feed(self, input_value_list, const_value_list):
            # ...
    
        # clear flags previously in the DAG
        def clear(self):
            # ...
    
        # compute forward
        def compute_forward(self):
            # ...
    
        # back prop
        def back_prop(self):
            # ...

nn class

    # higher level API for Neural Networks
    class nn:
    
        def __init__(self):
            return
    
        # for y = \sum_i (x_i), return y
        def scalar_sum(self, n, x):
            # ...
    
        # for a_{n} + b_{n} = c_{n}, return c
        def vector_add(self, n, a, b):
            # ...
    
        # for W_{m*n} * x_{n*1} = y_{m*1}, where W has been flattened
        # firstly according to array, W, x are all nodes;
        # this function will return y, which is a list of m nodes
        def matrix_multiply(self, m, n, W, x):
            # ...
    
        # fully connected layer, W_{m*n} * x_{n*1} + b_{m*1} = y_{m*1}
        # this function will return [y, W, b]
        def FC(self, input_size, output_size, x, weights = None, bias = None):
            # ...
    
        # ReLU layer, input size: n; this function will return y
        def ReLU(self, input_size, x):
            # ...
    
        # Convolution layer, this version supports a 3 by 3 by 1 (Channels = 1) Filter,
        # and a Same Padding which retains the size of the Input Image x_{m*n}
        # this function will return [y, W, b, zero], where zero is list of one node
        # which is the Constant 0 Padding node, and should be consider as a const node
        # and fed with 0 before running the network
        def Conv(self, m, n, x, weights = None, bias = None, padding_zero = None):
            # ...
    
        # Softmax layer, return y
        def Softmax(self, n, x):
            # ...
    
        # Cross-Entropy loss, return loss node
        def Cross_Entropy_Loss(self, n, y_pred, y_label_one_hot):
            # ...
    
        # ...

Hope you like the work!


