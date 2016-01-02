"""Implementation of logistic regression. Used as a baseline."""
from util import *

class LogReg(object):
  def __init__(self, logreg_config):
    self.name_ = logreg_config.name
    self.num_dims_    = logreg_config.num_inputs
    self.num_outputs_ = logreg_config.num_outputs
    self.w_ = Param((self.num_outputs_, self.num_dims_), logreg_config.w)
    self.b_ = Param((1, self.num_outputs_), logreg_config.b)
    self.param_list_ = [
      ('%s:w' % self.name_, self.w_),
      ('%s:b' % self.name_, self.b_),
    ]
    self.dropprob_ = logreg_config.dropprob

  def __str__(self):
    return 'W: %s \n b : %s' % (self.w_.__str__(), self.b_.__str__())

  def Load(self, f):
    for name, p in self.param_list_:
      p.Load(f, name)

  def Save(self, f):
    for name, p in self.param_list_:
      p.Save(f, name)

  def SetBatchSize(self, batch_size):
    self.o_ = cm.empty((batch_size, self.num_outputs_))
    self.o_deriv_ = cm.empty((batch_size, self.num_outputs_))
    self.c_ = cm.empty((batch_size, 1))
    #self.c_ = cm.empty((batch_size, self.num_outputs_))

  def Fprop(self, data, train=False):
    if self.dropprob_ and train > 0:
      data.dropout(self.dropprob_, scale=1.0/(1-self.dropprob_))
    cm.dot(data, self.w_.GetW().T, target=self.o_)
    self.o_.add_row_vec(self.b_.GetW())
    self.o_.apply_softmax_row_major()

  def GetPredictions(self):
    return self.o_

  def ComputeDeriv(self, t):
    self.o_.apply_softmax_grad_row_major(t, target=self.o_deriv_)
    #self.o_deriv_.assign(1)

  def Bprop(self, deriv):
    cm.dot(self.o_deriv_, self.w_.GetW(), target=deriv)

  def Outp(self, data):
    cm.dot(self.o_deriv_.T, data, target=self.w_.GetdW())
    self.o_deriv_.sum(axis=0, target=self.b_.GetdW())

  def Update(self):
    self.w_.Update()
    self.b_.Update()

  def GetCorrect(self, t):
    self.o_.get_softmax_correct_row_major(t, self.c_)
    return self.c_.sum()

  def GetLoss(self, t):
    """ Linear loss for grad check."""
    self.o_.subtract(t, target=self.c_)
    return self.c_.sum()

  def GetParams(self):
    return self.param_list_

  def GetOutputDims(self):
    return self.num_outputs_
