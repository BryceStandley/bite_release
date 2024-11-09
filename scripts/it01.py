# This is a simple script to run the full inference pipeline including the TTOPT loss computation without large argument strings in the command line.

import full_inference_including_ttopt

model_file_complete = 'cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar'
config = 'refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml'
suffix = 'custom'
workers = 4
loss_weight_ttopt_path = 'bite_loss_weights_ttopt.json'

class bite_args:
    model_file_complete = ''
    config = ''
    suffix = ''
    workers = 0
    loss_weight_ttopt_path = ''

args = bite_args()
args.model_file_complete = model_file_complete
args.config = config
args.suffix = suffix
args.workers = workers
args.loss_weight_ttopt_path = loss_weight_ttopt_path

full_inference_including_ttopt.main(args)