from train import train_from_config, usage
import sys

if __name__ == '__main__':
   skip_training = True
   if (len(sys.argv) == 2):
     train_from_config(config_path=sys.argv[1], skip_training=skip_training)
   elif len(sys.argv) == 5:
     train_from_config(
           train_path=sys.argv[1],
           dev_path=sys.argv[2],
           config_path=sys.argv[3],
           model_path=sys.argv[4], 
           skip_training=skip_training)
   else:
      usage()
