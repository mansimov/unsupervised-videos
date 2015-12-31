"""Displays reconstruction and future predictions for trained models."""

from data_handler import *
from lstm_combo import *
import lstm

def main():
  model = ReadModelProto(sys.argv[1])
  lstm_autoencoder = LSTMCombo(model)
  data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
  lstm_autoencoder.Show(data, output_dir='./imgs/mnist_1layer_example.pdf')

if __name__ == '__main__':
  # Set the board
  board_id = int(sys.argv[3])
  board = LockGPU(board=board_id)
  print 'Using board', board

  cm.CUDAMatrix.init_random(42)
  np.random.seed(42)
  main()
