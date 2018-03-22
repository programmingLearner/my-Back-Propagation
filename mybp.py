# Tianyang Zhao

import numpy as np
from queue import Queue

# error handling function
def error(warning):
    print(warning)
    exit()

# progress reporting function, may be disabled after debugging
def progress(report):
    # print(report)
    return

glob = 0

# computational node in DAG, similar to placeholder of Tensor Flow
class node:

    # constructor
    def __init__(self, operator, left_child, right_child = None, is_output = False):
        # global  glob
        # glob += 1
        # declaration of variables of this class
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
        # initiate
        self.check_valid_op(operator)
        self.operator = operator
        self.left_child = left_child
        self.right_child = right_child
        if operator in {'input', 'const'}:
            return
        if self.left_child == None:
            error('Error: invalid operator without child!')
        self.left_child.parents.append(self)
        self.left_child.cnt_parents += 1
        if operator in {'+', '*'}:  # binary operator
            if self.right_child == None:
                error('Error: invalid binary operator without child!')
            self.right_child.parents.append(self)
            self.right_child.cnt_parents += 1

    # set as output node
    def set_as_output(self):
        self.is_output = True

    # clear flags previously in the nodes
    def clear(self):
        # global glob
        # glob -= 1
        self.forward = 0
        self.backward = 0
        self.cnt_ref = 0
        self.init = False
        self.bp_ed = False

    def check_valid_op(self, operator):
        if operator not in {'input', 'const', '+', '*', 'sin', 'cos', 'tan',
                            'exp', 'log', 'neg', 'inv', 'relu'}:
            error('Error: invalid operator type: ' + operator + ' !')

    def check_initiated(self):
        if self.init == False:
            error('Error: attempting to run graph without initializing!')

    # feed value, initiate
    def feed(self, value):
        if self.operator not in {'input', 'const'}:
            error('Error: attempting to feed value to non-input-or-const node!')
        self.forward = value

    # compute forwardly: compute the value of this node forwarding
    # this function will return the node's parents nodes who is ready to compute forwardly
    def compute_forward(self):
        # valid check
        if self.operator not in {'input', 'const'}:
            if self.init == True:
                error('Error: attempting to recalculate forwardly.')
        # compute
        if self.operator in {'+', '*'}: # binary operator
            self.left_child.check_initiated()
            self.right_child.check_initiated()
            if self.operator == '+':
                self.forward = self.left_child.forward + self.right_child.forward
            else:
                self.forward = self.left_child.forward * self.right_child.forward
        elif self.operator not in {'input', 'const'}:   # unary operator
            self.left_child.check_initiated()
            if self.operator == 'sin':
                self.forward = np.sin(self.left_child.forward)
            elif self.operator == 'cos':
                self.forward = np.cos(self.left_child.forward)
            elif self.operator == 'tan':
                self.forward = np.tan(self.left_child.forward)
            elif self.operator == 'exp':
                self.forward = np.exp(self.left_child.forward)
            elif self.operator == 'log':
                if self.left_child.forward <= 0:
                    error('Error: attempting to feed a non-positive number into log!')
                self.forward = np.log(self.left_child.forward)
            elif self.operator == 'neg':
                self.forward = 0 - self.left_child.forward
            elif self.operator == 'inv':
                if self.left_child.forward == 0:
                    error('Error: attempting to feed 0 into inverse!')
                self.forward = 1.0 / self.left_child.forward
            elif self.operator == 'relu':
                if self.left_child.forward > 0:
                    self.forward = self.left_child.forward
                else:
                    self.forward = 0
            else:
                error('Error: invalid operator type while forwarding!')
        self.init = True    # this should be in advance of find parents,
                            # in case, e.g. b = a + a
        # find parents ready for forwarding
        list = []
        tmp = None
        for node in self.parents:
            if tmp == node: # in case like b = a + a
                continue
            tmp = node
            if node.operator in {'+', '*'}:
                if node.left_child == self:
                    if node.right_child.init == True:
                        list.append(node)
                else:
                    if node.left_child.init == True:
                        list.append(node)
            else:
                list.append(node)
        return list

    # computing gradient w.r.t the child of this node, adding it to that
    # this function will return the node's children nodes who is ready to back-prop
    def back_prop(self):
        # special cases
        if self.is_output == True:
            if self.init == False:
                error('Error: attempting to back-prop without forwarding in advance!')
            self.backward = 1
        if self.bp_ed == True:
            error('Error: attempting to back-prop twice from the node!')
        if self.cnt_parents != self.cnt_ref:
            error('Error: attempting to back-prop in a wrong order!')
        if self.operator in {'input', 'const'}:
            self.bp_ed = True
            return []
        # compute
        if self.operator in {'+', '*'}:
            if self.operator == '+':
                self.left_child.backward += self.backward
                self.right_child.backward += self.backward
            else:
                self.left_child.backward += self.backward * self.right_child.forward
                self.right_child.backward += self.backward * self.left_child.forward
            self.left_child.cnt_ref += 1
            self.right_child.cnt_ref += 1
        else:
            if self.operator == 'sin':
                self.left_child.backward += self.backward * np.cos(self.left_child.forward)
            elif self.operator == 'cos':
                self.left_child.backward -= self.backward * np.sin(self.left_child.forward)
            elif self.operator == 'tan':
                self.left_child.backward += self.backward / ((np.cos(self.left_child.forward)) ** 2)
            elif self.operator == 'exp':
                self.left_child.backward += self.backward * np.exp(self.left_child.forward)
            elif self.operator == 'log':
                self.left_child.backward += self.backward / self.left_child.forward
            elif self.operator == 'neg':
                self.left_child.backward -= self.backward
            elif self.operator == 'inv':
                self.left_child.backward -= self.backward / (self.left_child.forward ** 2)
            elif self.operator == 'relu':
                if self.left_child.forward > 0:
                    self.left_child.backward += self.backward
            else:
                error('Error: invalid operator type while back-prop!')
            self.left_child.cnt_ref += 1
        self.bp_ed = True
        # find children who are ready to back-prop: (considering b = a + a)
        list = []
        if self.operator in {'+', '*'}:
            if self.left_child.cnt_ref == self.left_child.cnt_parents:
                list.append(self.left_child)
            if self.right_child.cnt_ref == self.right_child.cnt_parents:
                if self.right_child != self.left_child:
                    list.append(self.right_child)
        else:
            if self.left_child.cnt_ref == self.left_child.cnt_parents:
                list.append(self.left_child)
        return list


# computational graph presented as DAG
class graph:

    # constructor
    def __init__(self, input_nodes, const_nodes, output_nodes):
        progress('Welcome to use zty automatic back-prop program, ' +
              'make sure that you have indicated which unique node is the output.')
        self.input_nodes = input_nodes
        self.const_nodes = const_nodes
        self.output_nodes = output_nodes
        self.visited_ind = 1    # the nodes in the graph is visited during a traversing
                                # if its visited bit is the same as self.visited indicator
        progress('Progress: computational graph imported.')

    # feed value, initiate
    def feed(self, input_value_list, const_value_list):
        l = len(input_value_list)
        if l != len(self.input_nodes):
            error('Error: input value list does not match!')
        for i in range(l):
            self.input_nodes[i].feed(input_value_list[i])
        l = len(const_value_list)
        if l != len(self.const_nodes):
            error('Error: const value list does not match!')
        for i in range(l):
            self.const_nodes[i].feed(const_value_list[i])
        progress('Progress: computational graph fed.')

    # clear flags previously in the DAG
    def clear(self):
        queue = Queue()
        for elm in self.output_nodes:
            queue.put(elm)
            elm.visited = self.visited_ind
        while not queue.empty():
            # print(queue.queue)
            node = queue.get()
            node.clear()
            if node.left_child is not None:
                if node.left_child.visited != self.visited_ind:
                    queue.put(node.left_child)
                    node.left_child.visited = self.visited_ind
            if node.right_child is not None:
                if node.right_child.visited != self.visited_ind:
                    queue.put(node.right_child)
                    node.right_child.visited = self.visited_ind
        self.visited_ind = 1 - self.visited_ind
        progress('Progress: computational graph cleared.')

    # compute forward
    def compute_forward(self):
        queue = Queue()
        for elm in self.input_nodes:
            queue.put(elm)
        for elm in self.const_nodes:
            queue.put(elm)
        while not queue.empty():
            node = queue.get()
            preparing = node.compute_forward()
            for elm in preparing:
                queue.put(elm)
        progress('Progress: computational graph forwarded.')

    # back prop
    def back_prop(self):
        queue = Queue()
        for elm in self.output_nodes:
            queue.put(elm)
        while not queue.empty():
            # print(queue.queue)
            node = queue.get()
            preparing = node.back_prop()
            for elm in preparing:
                queue.put(elm)
        progress('Progress: computational graph back-propagated.')


# higher level API for Neural Networks
class nn:

    def __init__(self):
        return

    # for y = \sum_i (x_i), return y
    def scalar_sum(self, n, x):
        if n <= 0:
            error('Error: scalar sum exception!')
        if n == 1:
            return x[0]
        sum = node('+', x[0], x[1])
        j = 2   # not 3
        while j < n:
            sum = node('+', sum, x[j])
            j += 1
        return sum

    # for a_{n} + b_{n} = c_{n}, return c
    def vector_add(self, n, a, b):
        c = []
        for i in range(n):
            c.append(node('+', a[i], b[i]))
        return c

    # for W_{m*n} * x_{n*1} = y_{m*1}, where W has been flattened
    # firstly according to array, W, x are all nodes;
    # this function will return y, which is a list of m nodes
    def matrix_multiply(self, m, n, W, x):
        y = []
        for i in range(m):
            if n == 1:
                sum = node('*', W[i], x[0])
                y.append(sum)
                continue
            sum = node('+', node('*', W[i*n], x[0]), node('*', W[i*n+1], x[1]))
            j = 2   # not 3 !!!!!
            while j < n:
                product = node('*', W[i * n + j], x[j])
                sum = node('+', sum, product)
                j += 1
            y.append(sum)
        return y

    # fully connected layer, W_{m*n} * x_{n*1} + b_{m*1} = y_{m*1}
    # this function will return [y, W, b]
    def FC(self, input_size, output_size, x, weights = None, bias = None):
        m = output_size
        n = input_size
        W = []
        b = []
        if weights == None:
            for i in range(m):
                for j in range(n):
                    tmp = node('input', None)
                    W.append(tmp)
        else:
            W = weights
        if bias == None:
            for i in range(m):
                tmp = node('input', None)
                b.append(tmp)
        else:
            b = bias
        return [self.vector_add(m, b, self.matrix_multiply(m, n, W, x)), W, b]

    # ReLU layer, input size: n; this function will return y
    def ReLU(self, input_size, x):
        y = []
        for i in range(input_size):
            tmp = node('relu', x[i])
            y.append(tmp)
        return y

    # Convolution layer, this version supports a 3 by 3 by 1 (Channels = 1) Filter,
    # and a Same Padding which retains the size of the Input Image x_{m*n}
    # this function will return [y, W, b, zero], where zero is list of one node
    # which is the Constant 0 Padding node, and should be consider as a const node
    # and fed with 0 before running the network
    def Conv(self, m, n, x, weights = None, bias = None, padding_zero = None):
        # Same Padding for 3 by 3 by 1 (1 Channel) Convolution Filter
        zero = []
        if padding_zero == None:
            zero.append(node('const', None))
        else:
            zero = padding_zero
        padded_x = []
        for i in range(m + 2):
            for j in range(n + 2):
                if (i == 0) or (i == m + 1) or (j == 0) or (j == n + 1):
                    padded_x.append(zero[0])
                else:
                    padded_x.append(x[(i - 1) * n + (j - 1)])
        # Initialize weights and bias
        W = []
        b = []
        if weights == None:
            for i in range(3 * 3):
                tmp = node('input', None)
                W.append(tmp)
        else:
            W = weights
        if bias == None:
            b.append(node('input', None))
        else:
            b = bias
        # Convolution
        y = []
        for i in range(m):
            for j in range(n):
                Conv_input = []
                for p in range(3):
                    for q in range(3):
                        Conv_input.append(padded_x[(i + p) * (n + 2) + (j + q)])
                Conv_output = self.FC(3 * 3, 1, Conv_input, W, b)[0]
                y.append(Conv_output[0])
        return([y, W, b, zero])

    # Softmax layer, return y
    def Softmax(self, n, x):
        y = []
        exps = []
        for i in range(n):
            exps.append(node('exp', x[i]))
        sum = self.scalar_sum(n, exps)
        inv_sum = node('inv', sum)
        for i in range(n):
            y.append(node('*', exps[i], inv_sum))
        return y

    # Cross-Entropy loss, return loss node
    def Cross_Entropy_Loss(self, n, y_pred, y_label_one_hot):
        list = []
        for i in range(n):
            tmp = node('log', y_pred[i])
            tmp2 = node('*', tmp, y_label_one_hot[i])
            list.append(tmp2)
        loss = self.scalar_sum(n, list)
        return node('neg', loss)

    # return a list of input nodes
    def new_input_nodes_list(self, n):
        input = []
        for i in range(n):
            input.append(node('input', None))
        return input

    # return a list of const nodes
    def new_const_nodes_list(self, n):
        const = []
        for i in range(n):
            const.append(node('const', None))
        return const


#############################################################################
# An example
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

# compute
G = graph([x1,x2,x3], [c1,c2,c3], [o])
x_init = np.random.random(size = 3)
const_init = [1.0, 2.0, 1.0]
G.feed(x_init, const_init)
G.compute_forward()
G.back_prop()
yo = o.forward
gradient = [x1.backward, x2.backward, x3.backward]
print('')
print('Gradient w.r.t. x1, x2, x3 are: ')
print(gradient)

# validate
delta = 0.001
numerical_gradient = [0.0, 0.0, 0.0]
for i in range(3):
    # note the shallow copy problem of python!
    x_init_delta = [0.0, 0.0, 0.0]
    for j in range(len(x_init_delta)):
        x_init_delta[j] = x_init[j]
    x_init_delta[i] += delta
    G.clear()
    G.feed(x_init_delta, const_init)
    G.compute_forward()
    numerical_gradient[i] = (o.forward - yo) / delta
print('Numerical gradient w.r.t. x1, x2, x3 are: ')
print(numerical_gradient)
print('')
G.clear()

##############################################################################
# An example
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

# initialize
init_weights = np.random.random(size = Conv_size + 1 + Conv_size + 1
                                + size * size * FC_nodes + FC_nodes
                                + FC_nodes * classes + classes) - 0.5
init_image = np.random.random(size = size * size)
init_label = [0,1,0,0,0,0,0,0,0,0]
init_zero = [0]

Graph.feed(list(init_weights), list(init_image) + init_label + init_zero)
print('')

# compute
Graph.compute_forward()
Graph.back_prop()

print('Cross-entropy loss of ResNet is:')
v1 = Loss.forward
print(v1)
print('Total parameters number is:')
print(params)

Gradients = []
for item in Weights:
    Gradients.append(item.backward)

# validate
difference = np.random.random(size = params)
t = 0.0001
init_weights_2 = []
for i in range(params):
    init_weights_2.append(t * difference[i] + init_weights[i])
Graph.clear()
Graph.feed(list(init_weights_2), list(init_image) + init_label + init_zero)
Graph.compute_forward()
v2 = Loss.forward

left = (v2 - v1) / t
right = np.dot(Gradients, difference)
print('Validation:')
print('left = ')
print(left)
print('right = ')
print(right)

##############################################################################