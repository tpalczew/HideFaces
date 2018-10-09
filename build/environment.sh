
export BASE_DIR=`pwd`
export PYTHONPATH=$BASE_DIR/src/ingestion/:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/src/model/:$PYTHONPATH

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
