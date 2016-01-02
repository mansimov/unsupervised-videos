from data_handler import *
import lstm
import datetime

class LSTMClassifier(object):
  def __init__(self, model):
    self.model_ = model
    self.lstm_stack_ = lstm.LSTMStack()
    for l in model.lstm:
      self.lstm_stack_.Add(lstm.LSTM(l))
    self.squash_relu_ = model.squash_relu
    self.squash_relu_lambda_ = model.squash_relu_lambda
    
    if len(model.timestamp) > 0:
      old_st = model.timestamp[-1]
      ckpt = os.path.join(model.checkpoint_dir, '%s_%s.h5' % (model.name, old_st))
      f = h5py.File(ckpt)
      self.lstm_stack_.Load(f)
      f.close()

  # used to check if gradient fucntion was implemented correctly
  def GradCheck(self):
    eps = 0.01
    tol = 1e-3
    params_list = []
    params_list.extend(self.lstm_stack_.GetParams())
    self.num_dims_ = self.lstm_stack_.GetInputDims()
    self.num_output_dims_ = self.lstm_stack_.GetOutputDims()
    self.SetBatchSize(128, 3)
    v_cpu = np.random.randn(self.batch_size_, self.seq_length_ * self.num_dims_)
    t_cpu = np.random.randn(self.batch_size_, self.num_output_dims_)
    self.v_.overwrite(v_cpu)
    self.target_.overwrite(t_cpu)
    
    self.Fprop()
    self.BpropAndOutp()
    
    for name, param in params_list:
      print name
      w = param.GetW()
      dw = param.GetdW()
      fail = False
      for row in xrange(min(4, w.shape[0])):
        for col in xrange(min(4, w.shape[1])):
          val = w.read_value(row, col)
          w.write_value(row, col, val+eps)
          self.Fprop()
          l1 = self.cl_.GetLoss(self.target_)
          w.write_value(row, col, val - eps)
          self.Fprop()
          l2 = self.cl_.GetLoss(self.target_)
          grad_n = (l1 - l2 ) / (2 * eps)
          grad_a = dw.read_value(row, col)
          diff = np.abs(grad_n - grad_a) / (np.abs(grad_n) + np.abs(grad_a))
          print 'Numerical %.8f Analytical %.8f Diff %.5f' % (grad_n, grad_a, diff)
          if diff > tol:
            fail = True
          w.write_value(row, col, val)
      if fail:
        res = 'FAILED'
      else:
        res= 'PASSED'
      print res

  def Fprop(self, train=False):
    if self.squash_relu_:
      self.v_.apply_relu_squash(lambdaa=self.squash_relu_lambda_)
    num_models = self.lstm_stack_.GetNumModels()
    self.lstm_stack_.Reset()
    for t in xrange(self.seq_length_):
      # slice input and output at timestep t and get probabilities
      i = self.v_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      o = self.o_.col_slice(t * self.num_output_dims_, (t+1) * self.num_output_dims_)
      self.lstm_stack_.Fprop(input_frame=i, output_frame=o, train=train)
      o.apply_softmax_row_major()

  # compute derivative only for softmax
  def ComputeDeriv(self):
    for t in xrange(self.seq_length_):
      o = self.o_.col_slice(t * self.num_output_dims_, (t+1) * self.num_output_dims_)
      o_deriv = self.o_deriv_.col_slice(t * self.num_output_dims_, (t+1) * self.num_output_dims_)
      o.apply_softmax_grad_row_major(self.target_, target=o_deriv)

  def GetLoss(self):
    batch_size = self.o_.shape[0]
    self.o_.reshape((-1, self.seq_length_))
    self.avg_o_.reshape((-1, 1))
    self.o_.sum(axis=1, target=self.avg_o_)
    self.avg_o_.mult(1.0 / self.seq_length_)
    self.o_.reshape((batch_size, -1))
    self.avg_o_.reshape((batch_size, -1))
    self.avg_o_.get_softmax_correct_row_major(self.target_, self.c_)
    return self.c_.sum()

  def GetPrediction(self):
    batch_size = self.o_.shape[0]
    self.o_.reshape((-1, self.seq_length_))
    self.avg_o_.reshape((-1, 1))
    self.o_.sum(axis=1, target=self.avg_o_)
    self.avg_o_.mult(1.0 / self.seq_length_)
    self.o_.reshape((batch_size, -1))
    self.avg_o_.reshape((batch_size, -1))
    return self.avg_o_

  def BpropAndOutp(self):
    num_models = self.lstm_stack_.GetNumModels()
    for t in xrange(self.seq_length_-1, -1, -1):
      i = self.v_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      o_deriv = self.o_deriv_.col_slice(t * self.num_output_dims_, (t+1) * self.num_output_dims_)
      self.lstm_stack_.BpropAndOutp(input_frame=i, output_deriv=o_deriv)

  def Update(self):
    self.lstm_stack_.Update()

  def Validate(self, data):
    data.Reset()
    dataset_size = data.GetDatasetSize()
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    if dataset_size % batch_size > 0:
      num_batches += 1
    loss = 0
    preds = np.zeros((dataset_size, self.num_output_dims_), dtype=np.float32)
    start = 0
    for ii in xrange(num_batches):
      v_cpu, t_cpu = data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.target_.overwrite(t_cpu)
      self.Fprop()
      end = min(start + batch_size, dataset_size)
      preds[start:end, :] = self.GetPrediction().asarray()[:end-start,:]
      start = end
    correct, pooled_correct = data.GetResults(preds)
    return correct, pooled_correct

  # Note that both train and valid should have the same batch_size
  def SetBatchSize(self, batch_size, seq_length):
    self.batch_size_ = batch_size
    self.seq_length_ = seq_length
    self.lstm_stack_.SetBatchSize(batch_size, seq_length)
    self.v_ = cm.empty((batch_size, seq_length * self.num_dims_))
    self.o_ = cm.empty((batch_size, seq_length * self.num_output_dims_))
    self.o_deriv_ = cm.empty((batch_size, seq_length * self.num_output_dims_))
    self.avg_o_ = cm.empty((batch_size, self.num_output_dims_))
    self.target_ = cm.empty((batch_size, 1))
    self.c_ = cm.empty((batch_size, 1))

  def Save(self, model_file):
    sys.stdout.write(' Writing model to %s' % model_file)
    f = h5py.File(model_file, 'w')
    self.lstm_stack_.Save(f)
    f.close()

  def Train(self, train_data, valid_data=None):
    # Timestamp the model that we are training.
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_file = os.path.join(self.model_.checkpoint_dir, '%s_%s' % (self.model_.name, st))
    self.model_.timestamp.append(st)
    print 'Model saved at %s.pbtxt' % model_file
    WritePbtxt(self.model_, '%s.pbtxt' % model_file)
   
    self.num_dims_ = self.lstm_stack_.GetInputDims()
    self.num_output_dims_ = self.lstm_stack_.GetOutputDims()
    batch_size = train_data.GetBatchSize()
    seq_length = train_data.GetSeqLength()

    self.SetBatchSize(batch_size, seq_length)

    loss = 0
    temp_loss = loss
    best_val_loss = False
    print_after = self.model_.print_after
    validate_after = self.model_.validate_after
    validate = validate_after > 0 and valid_data is not None
    save_after = self.model_.save_after
    save = save_after > 0
    display_after = self.model_.display_after
    display = display_after > 0
    temp_valid_loss = 0

    for ii in xrange(1, self.model_.max_iters + 1):
      newline = False
      sys.stdout.write('\rStep %d' % ii)
      sys.stdout.flush()

      v_cpu, t_cpu = train_data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.target_.overwrite(t_cpu)

      self.Fprop(train=True)

      # Compute Performance.
      loss += self.GetLoss() / batch_size

      if ii % print_after == 0:
        loss /= print_after
        sys.stdout.write(' Acc %.5f' % loss)
        temp_loss = loss
        loss = 0
        newline = True

      # compute derivatives for softmax -> compute derivatives for lstm layers
      self.ComputeDeriv()
      self.BpropAndOutp()
      self.Update()

      if display and ii % display_after == 0:
        self.lstm_stack_.Display()

      if validate and ii % validate_after == 0:
        valid_loss, valid_loss_pooled = self.Validate(valid_data)
        if valid_loss_pooled > temp_valid_loss:
          best_val_loss = True
          temp_valid_loss = valid_loss_pooled
        else:
          best_val_loss = False
        temp_loss = 0
        sys.stdout.write(' Valid Acc %.5f ; Pooled Valid Acc %.5f' % (valid_loss, valid_loss_pooled))
        newline = True

      if save and ii % save_after == 0:
        self.Save('%s.h5' % model_file)
      if save and best_val_loss == True:
        self.Save('%s_best.h5' % model_file)
        best_val_loss = False
      if newline:
        sys.stdout.write('\n')

    sys.stdout.write('\n')

def main():
  model = ReadModelProto(sys.argv[1])
  lstm_classifier = LSTMClassifier(model)
  train_data = DataHandler(ReadDataProto(sys.argv[2]))
  valid_data = DataHandler(ReadDataProto(sys.argv[3]))
  lstm_classifier.Train(train_data, valid_data)

if __name__ == '__main__':
  board_id = int(sys.argv[4])
  board = LockGPU(board=board_id)
  print 'Using board', board
  
  cm.CUDAMatrix.init_random(42)
  np.random.seed(42)
  main()
