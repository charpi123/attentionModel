import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class attentionModel(object):
	def __init__(self,nh, nc, ne, de):
		'''
		nh :: dimension of the hidden layer
		nc :: number of classes
		ne :: number of word embeddings in the vocabulary
		de :: dimension of the word embeddings
		'''

		# parameters of the model
		self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(de,ne)).astype(theano.config.floatX))
		self.emby = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(de,ne)).astype(theano.config.floatX))

		# encoder
		# forward parameters
		# W parameters
		self.forwardW = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.forwardWz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.forwardWr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))

		# U parameters
		self.forwardU = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.forwardUz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.forwardUr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))

		# backward parameters
		# W parameters
		self.backwardW = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.backwardWz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.backwardWr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))

		# U parameters
		self.backwardU = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.backwardUz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.backwardUr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))

		# constant zeros
		self.forwardh0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
		self.backwardh0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

		# decoder
		self.fW = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.fWz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))
		self.fWr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,de)).astype(theano.config.floatX))

		self.fU = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.fUz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.fUr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))

		self.fC = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,2*nh)).astype(theano.config.floatX))
		self.fCz = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,2*nh)).astype(theano.config.floatX))
		self.fCr = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,2*nh)).astype(theano.config.floatX))

		self.va = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh)).astype(theano.config.floatX))
		self.Wa = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))
		self.Ua = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,2*nh)).astype(theano.config.floatX))
		self.Ws = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nh,nh)).astype(theano.config.floatX))

		# output
		self.Wo = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(nc,nh)).astype(theano.config.floatX))
		self.Uo = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(2*nh,nh)).astype(theano.config.floatX))
		self.Vo = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(2*nh,nh)).astype(theano.config.floatX))
		self.Co = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(2*nh,2*nh)).astype(theano.config.floatX))

		self.maxOut = theano.shared(0.2 * numpy.random.uniform(-1.0,1.0,(2*nh,nh)).astype(theano.config.floatX))

		# bundle
		self.params = [self.emb, self.emby, self.forwardW, self.forwardWz, self.forwardWr, self.forwardU, self.forwardUz, self.forwardUr, self.backwardW, self.backwardWz, self.backwardWr, self.backwardU, self.backwardUz, self.backwardUr, self.fW, self.fWz, self.fWr, self.fU, self.fUz, self.fUr, self.fC, self.fCz, self.fCr, self.va, self.Wa, self.Ua, self.Ws, self.forwardh0, self.backwardh0, self.Wo, self.Uo, self.Vo, self.Co, self.maxOut]

		x_e = T.imatrix()
		x_c = T.imatrix()

		def forwardRecurrence(x_t,h_tm1):
			ri = T.nnet.sigmoid(T.dot(self.forwardWr,T.dot(self.emb,x_t)) + T.dot(self.forwardUr,h_tm1))
			zi = T.nnet.sigmoid(T.dot(self.forwardWz,T.dot(self.emb,x_t)) + T.dot(self.forwardUz,h_tm1))
			hi = T.tanh(T.dot(self.forwardW,T.dot(self.emb,x_t)) + T.dot(self.forwardU,(ri*h_tm1)))
			ho = (1-zi)*h_tm1 + zi*hi
			return ho

		def backwardRecurrence(x_t,h_tm1):
			ri = T.nnet.sigmoid(T.dot(self.backwardWr,T.dot(self.emb,x_t)) + T.dot(self.backwardUr,h_tm1))
			zi = T.nnet.sigmoid(T.dot(self.backwardWz,T.dot(self.emb,x_t)) + T.dot(self.backwardUz,h_tm1))
			hi = T.tanh(T.dot(self.backwardW,T.dot(self.emb,x_t)) + T.dot(self.backwardU,(ri*h_tm1)))
			ho = (1-zi)*h_tm1 + zi*hi
			return ho

		forwardH,_ = theano.scan(fn=forwardRecurrence, sequences=x_e, outputs_info=self.forwardh0, n_steps=x_e.shape[0])
		backwardH,_ = theano.scan(fn=backwardRecurrence, sequences=x_e[::-1], outputs_info=self.backwardh0, n_steps=x_e.shape[0])

		finalH = T.concatenate([forwardH,backwardH[::-1]],axis=1) # Tx by 2*nh dimensions
		
		def decoder(y_t,s_tm1,hid):
			'''
			y_t: 1 by vocab(chinese)
			s_tm1: nh by 1
			hid: Tx by nh*2
			'''
			
			eij,u = theano.scan(fn=lambda h_j, stm1:T.dot(self.va.T,T.tanh(T.dot(self.Wa,stm1) + T.dot(self.Ua,h_j))), sequences=hid, non_sequences=s_tm1, outputs_info=None) # eij is of length Tx (length of english sentence)

			#eij = T.dot(self.va,T.tanh(T.dot(self.Wa,s_tm1) + T.dot(self.Ua,hid.T))) # shawn's implementation

			aij = T.nnet.softmax(eij)[0] # aij is of length Tx
			
			ci = T.dot(aij,hid) # ci is of length 2*nh dimensions, shouldn't this be the sum?
			ri = T.nnet.sigmoid(T.dot(self.fWr,T.dot(self.emby,y_t)) + T.dot(self.fUr,s_tm1) + T.dot(self.fCr,ci))
			zi = T.nnet.sigmoid(T.dot(self.fWz,T.dot(self.emby,y_t)) + T.dot(self.fUz,s_tm1) + T.dot(self.fCz,ci))
			sbar = T.tanh(T.dot(self.emby,y_t) + T.dot(self.fU,(ri*s_tm1)) + T.dot(self.fC,ci))
			si = (1-zi)*s_tm1 + zi*sbar
			
			return si,ci
			#return [y_t.shape,s_tm1,s_tm1.shape,hid.shape]

		[rsi,rci],_ = theano.scan(fn=decoder, sequences=x_c, non_sequences=finalH, outputs_info=[T.tanh(T.dot(self.Ws,backwardH[0])),None])

		tbar,_ = theano.scan(fn=lambda sm1,ym1,ci: T.dot(self.Uo,sm1) + T.dot(self.Vo,T.dot(self.emby,ym1)) + T.dot(self.Co,ci),sequences=[dict(input=rsi, taps=[-1]),dict(input=x_c, taps=[-1]),rci],outputs_info=None,non_sequences=None)

		# reshape matrix to Tx2xl tensor, then compare across the '2' axis and get T.max(axis=1).

		ti_temp = T.reshape(tbar,(tbar.shape[0],tbar.shape[1]/2,2))
		ti = T.max(ti_temp,axis=2)[0]

		self.train = theano.function(inputs=[x_e,x_c],outputs=[finalH,rsi,tbar,ti_temp,ti],on_unused_input='ignore')

"""

def concatenate(tensor_list, axis=0):
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
 
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

"""