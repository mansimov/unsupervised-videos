from util import *

# LSTM layer
class LSTM(object):
  def __init__(self, lstm_config):
    self.name_ = lstm_config.name
    num_lstms = lstm_config.num_hid
    assert num_lstms  > 0
    self.num_lstms_   = num_lstms
    self.has_input_   = lstm_config.has_input
    self.has_output_  = lstm_config.has_output
    self.input_dims_  = lstm_config.input_dims
    self.output_dims_ = lstm_config.output_dims
    self.use_relu_    = lstm_config.use_relu
    self.input_dropprob_  = lstm_config.input_dropprob
    self.output_dropprob_ = lstm_config.output_dropprob
    self.t_ = 0

    print num_lstms
    # diag are peephole connections from cell state to diffent gates
    self.w_dense_  = Param((4 * num_lstms, num_lstms), lstm_config.w_dense)
    self.w_diag_   = Param((1, 3 * num_lstms), lstm_config.w_diag)
    self.b_        = Param((1, 4 * num_lstms), lstm_config.b)
    
    self.param_list_ = [
      ('%s:w_dense' % self.name_, self.w_dense_),
      ('%s:w_diag'  % self.name_, self.w_diag_),
      ('%s:b'       % self.name_, self.b_),
    ]
    
    if self.has_input_:
      assert self.input_dims_ > 0
      self.w_input_ = Param((4 * num_lstms, self.input_dims_), lstm_config.w_input)
      self.param_list_.append(('%s:w_input' % self.name_, self.w_input_))
    if self.has_output_:
      assert self.output_dims_ > 0
      self.w_output_ = Param((self.output_dims_, num_lstms), lstm_config.w_output)
      self.param_list_.append(('%s:w_output' % self.name_, self.w_output_))
      self.b_output_ = Param((1, self.output_dims_), lstm_config.b_output)
      self.param_list_.append(('%s:b_output' % self.name_, self.b_output_))

  def HasInputs(self):
    return self.has_input_

  def HasOutputs(self):
    return self.has_output_

  def GetParams(self):
    return self.param_list_

  def SetBatchSize(self, batch_size, seq_length):
    assert batch_size > 0
    assert seq_length > 0
    self.batch_size_  = batch_size
    self.seq_length_  = seq_length
    self.state_ = [cm.empty((batch_size, 6 * self.num_lstms_)) for i in xrange(seq_length)]
    self.deriv_ = [cm.empty((batch_size, 6 * self.num_lstms_)) for i in xrange(seq_length)]

    # dropout mask
    if self.has_output_ and self.output_dropprob_ > 0:
      self.output_drop_mask_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_state_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_deriv_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]

    if self.has_input_ and self.input_dropprob_ > 0:
      self.input_drop_mask_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_state_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_deriv_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]

  def Load(self, f):
    for name, p in self.param_list_:
      p.Load(f, name)

  def Save(self, f):
    for name, p in self.param_list_:
      p.Save(f, name)

  def Fprop(self, input_frame=None, init_state=None, output_frame=None, train=False, copy_init_state=True):
    t = self.t_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    output_slice = self.state_[t]
    gates = output_slice.col_slice(2 * num_lstms, 6 * num_lstms)
    lstm_state_computed = False
    
    if t == 0:
      if init_state is None:
        input_slice = self.state_[0]
        input_slice.assign(0)
        init = True
      else:
        if copy_init_state:
          output_slice.assign(init_state)
          lstm_state_computed = True
        else:
          input_slice = init_state
        init = False
    else:
      input_slice = self.state_[t-1]
      init = False
    
    # input to LSTM
    if self.has_input_ and input_frame is not None and not lstm_state_computed:
      if self.input_dropprob_ > 0 and train:
        mask = self.input_drop_mask_[t]
        intermediate_state = self.input_intermediate_state_[t]
        mask.assign(1 - self.input_dropprob_)
        mask.sample_bernoulli()
        mask.mult(1.0 / (1 - self.input_dropprob_))
        input_frame.mult(mask, target=intermediate_state)
        cm.dot(intermediate_state, self.w_input_.GetW().T, target=gates)
      else:
        cm.dot(input_frame, self.w_input_.GetW().T, target=gates)
    
    # internal LSTM state computations
    if not lstm_state_computed:
      cm.lstm_fprop(input_slice, output_slice,
                    self.w_dense_.GetW(), self.w_diag_.GetW(), self.b_.GetW(),
                    use_relu=self.use_relu_, init=init)

    # LSTM to output
    if self.has_output_:
      assert output_frame is not None
      state = output_slice.col_slice(0, num_lstms)
      
      if self.output_dropprob_ > 0 and train:
        mask = self.output_drop_mask_[t]
        intermediate_state = self.output_intermediate_state_[t]
        mask.assign(1 - self.output_dropprob_)
        mask.sample_bernoulli()
        mask.mult(1.0 / (1 - self.output_dropprob_))
        state.mult(mask, target=intermediate_state)
        cm.dot(intermediate_state, self.w_output_.GetW().T, target=output_frame)
      else:
        cm.dot(state, self.w_output_.GetW().T, target=output_frame)
      output_frame.add_row_vec(self.b_output_.GetW())
    
    self.t_ += 1

  # Bprop for getting gradients
  # Outp for updating weights
  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_state=None, init_deriv=None, output_deriv=None, copy_init_state=True):
    self.t_ -= 1

    t = self.t_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    output_slice_h = self.state_[t]
    output_slice_d = self.deriv_[t]

    # set gradients to zero
    if t == self.seq_length_ - 1:
      if self.has_output_:
        self.w_output_.GetdW().assign(0)
        self.b_output_.GetdW().assign(0)
      if self.has_input_:
        self.w_input_.GetdW().assign(0)
      self.w_dense_.GetdW().assign(0)
      self.w_diag_.GetdW().assign(0)
      self.b_.GetdW().assign(0)
    
    if self.has_output_:
      assert output_deriv is not None  # If this lstm's output was used, it must get a deriv back.
      deriv = output_slice_d.col_slice(0, num_lstms)
      state = output_slice_h.col_slice(0, num_lstms)
      if self.output_dropprob_ > 0:
        mask = self.output_drop_mask_[t]
        intermediate_state = self.output_intermediate_state_[t]
        intermediate_deriv = self.output_intermediate_deriv_[t]
        cm.dot(output_deriv.T, intermediate_state, target=self.w_output_.GetdW(), scale_targets=1.0)
        cm.dot(output_deriv, self.w_output_.GetW(), target=intermediate_deriv, scale_targets=0.0)
        intermediate_deriv.mult(mask)
        deriv.add(intermediate_deriv)
      else:
        cm.dot(output_deriv.T, state, target=self.w_output_.GetdW(), scale_targets=1.0)
        cm.dot(output_deriv, self.w_output_.GetW(), target=deriv, scale_targets=1.0)
      self.b_output_.GetdW().add_sums(output_deriv, axis=0)

    deriv_computed = False
    if t == 0:
      if init_state is None:
        input_slice_h  = self.state_[0]
        input_slice_d  = self.deriv_[0]
        init = True
      else:
        if copy_init_state:
          init_deriv.assign(output_slice_d)
          deriv_computed = True
        else:
          input_slice_h  = init_state
          input_slice_d  = init_deriv
        init = False
    else:
      input_slice_h  = self.state_[t-1]
      input_slice_d  = self.deriv_[t-1]
      init = False

    if not deriv_computed:
      cm.lstm_bprop(input_slice_h, output_slice_h,
                    input_slice_d, output_slice_d,
                    self.w_dense_.GetW(), self.w_diag_.GetW(),
                    use_relu=self.use_relu_, init=init)
      cm.lstm_outp(input_slice_h, output_slice_h, output_slice_d,
                   self.w_dense_.GetdW(), self.w_diag_.GetdW(), self.b_.GetdW(),
                   init=init)

    gates_deriv = output_slice_d.col_slice(2 * num_lstms, 6 * num_lstms)

    if self.has_input_ and input_frame is not None and not deriv_computed:
      if self.input_dropprob_ > 0:
        intermediate_state = self.input_intermediate_state_[t]
        cm.dot(gates_deriv.T, intermediate_state, target=self.w_input_.GetdW(), scale_targets=1.0)
        if input_deriv is not None:  # If the caller has asked for the deriv wrt input to be computed, do it.
          mask = self.input_drop_mask_[t]
          intermediate_deriv = self.input_intermediate_deriv_[t]
          cm.dot(gates_deriv, self.w_input_.GetW(), target=intermediate_deriv, scale_targets=0.0)
          intermediate_deriv.mult(mask)
          input_deriv.add(intermediate_deriv)
      else:
        cm.dot(gates_deriv.T, input_frame, target=self.w_input_.GetdW(), scale_targets=1.0)
        if input_deriv is not None:  # If the caller has asked for the deriv wrt input to be computed, do it.
          cm.dot(gates_deriv, self.w_input_.GetW(), target=input_deriv, scale_targets=1.0)

  def GetCurrentState(self):
    return self.state_[self.t_ - 1]

  def GetCurrentHiddenState(self):
    return self.state_[self.t_ - 1].col_slice(0, self.num_lstms_)
  
  def GetCurrentDeriv(self):
    return self.deriv_[self.t_ - 1]

  def GetCurrentHiddenDeriv(self):
    return self.deriv_[self.t_ - 1].col_slice(0, self.num_lstms_)

  def Update(self):
    self.w_dense_.Update()
    self.w_diag_.Update()
    self.b_.Update()
    if self.has_input_:
      self.w_input_.Update()
    if self.has_output_:
      self.w_output_.Update()
      self.b_output_.Update()

  def Display(self, fig=1):
    plt.figure(2*fig)
    plt.clf()
    name = ['h', 'c', 'i', 'f', 'a', 'o']
    for i in xrange(self.seq_length_):
      state = self.state_[i].asarray()
      for j in xrange(6):
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+1)
        start = j * self.num_lstms_
        end = (j+1) * self.num_lstms_
        plt.imshow(state[:, start:end])
        _, labels = plt.xticks()
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        #plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+7)
        plt.hist(state[:, start:end].flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+13)
        plt.hist(state[:, start:end].mean(axis=0).flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        plt.title('%s:%.3f' % (name[j],state[:, start:end].mean()))

    plt.draw()
    
    plt.figure(2*fig+1)
    plt.clf()
    name = ['w_dense', 'w_diag', 'b', 'w_input']
    ws = [self.w_dense_, self.w_diag_, self.b_, self.w_input_]
    l = len(ws)
    for i in xrange(l):
      w = ws[i]
      plt.subplot(1, l, i+1)
      plt.hist(w.GetW().asarray().flatten(), 100)
      _, labels = plt.xticks()
      plt.setp(labels, rotation=45)
      plt.title(name[i])
    plt.draw()

  def Reset(self):
    self.t_ = 0
    for t in xrange(self.seq_length_):
      self.state_[t].assign(0)
      self.deriv_[t].assign(0)

  def GetInputDims(self):
    return self.input_dims_
  
  def GetOutputDims(self):
    return self.output_dims_

# LSTMStack is a stack of different lstm layers
class LSTMStack(object):
  def __init__(self):
    self.models_ = []
    self.num_models_ = 0

  def Add(self, model):
    self.models_.append(model)
    self.num_models_ += 1

  def Fprop(self, input_frame=None, init_state=[], output_frame=None, train=False, copy_init_state=True):
    num_models = self.num_models_
    num_init_state = len(init_state)
    assert num_init_state == 0 or num_init_state == num_models
    for m, model in enumerate(self.models_):
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_init_state   = init_state[m] if num_init_state > 0 else None
      this_output_frame = output_frame if m == num_models - 1 else None
      model.Fprop(input_frame=this_input_frame,
                  init_state=this_init_state,
                  output_frame=this_output_frame,
                  train=train, copy_init_state=copy_init_state)

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_state=[], init_deriv=[], output_deriv=None, copy_init_state=True):
    num_models = self.num_models_
    num_init_state = len(init_state)
    assert num_init_state == 0 or num_init_state == num_models
    for m in xrange(num_models-1, -1, -1):
      model = self.models_[m]
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_input_deriv  = input_deriv if m == 0 else self.models_[m-1].GetCurrentHiddenDeriv() 
      this_init_state   = init_state[m] if num_init_state > 0 else None
      this_init_deriv   = init_deriv[m] if num_init_state > 0 else None
      this_output_deriv = output_deriv if m == num_models - 1 else None
      model.BpropAndOutp(input_frame=this_input_frame,
                         input_deriv=this_input_deriv,
                         init_state=this_init_state,
                         init_deriv=this_init_deriv,
                         output_deriv=this_output_deriv,
                         copy_init_state=copy_init_state)

  def Reset(self):
    for model in self.models_:
      model.Reset()

  def Update(self):
    for model in self.models_:
      model.Update()

  def GetNumModels(self):
    return self.num_models_
  
  def SetBatchSize(self, batch_size, seq_length):
    for model in self.models_:
      model.SetBatchSize(batch_size, seq_length)

  def Save(self, f):
    for model in self.models_:
      model.Save(f)

  def Load(self, f):
    for model in self.models_:
      model.Load(f)

  def GetCurrentHiddenState(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetCurrentHiddenState()
    else:
      return None

  def GetCurrentCellState(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetCurrentCellState()
    else:
      return None

  def GetCurrentHiddenDeriv(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetCurrentHiddenDeriv()
    else:
      return None

  def GetCurrentCellDeriv(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetCurrentCellDeriv()
    else:
      return None

  def Display(self):
    for m, model in enumerate(self.models_):
      model.Display(m)

  def GetParams(self):
    params_list = []
    for model in self.models_:
      params_list.extend(model.GetParams())
    return params_list

  def HasInputs(self):
    if self.num_models_ > 0:
      return self.models_[0].HasInputs()
    else:
      return False
  
  def HasOutputs(self):
    if self.num_models_ > 0:
      return self.models_[-1].HasOutputs()
    else:
      return False

  def GetInputDims(self):
    if self.num_models_ > 0:
      return self.models_[0].GetInputDims()
    else:
      return 0
  
  def GetOutputDims(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetOutputDims()
    else:
      return 0

  def GetAllCurrentStates(self):
    return [m.GetCurrentState() for m in self.models_]
  
  def GetAllCurrentDerivs(self):
    return [m.GetCurrentDeriv() for m in self.models_]
