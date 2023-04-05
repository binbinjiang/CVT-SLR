# w/ AE
CUDA_VISIBLE_DEVICES=6 python run_demo.py --work-dir ./out_cvpr/cvtslt_2/ --config ./configs/cvtslt_eval_config.yaml --device 1 --load-weights ./trained_models/cvtslt_model_dev_19.87.pt --use_seqAE AE
# test 20.17 dev 19.87

# w/ VAE
CUDA_VISIBLE_DEVICES=6 python run_demo.py --work-dir ./out_cvpr/cvtslt_1/ --config ./configs/cvtslt_eval_config.yaml --device 1 --load-weights ./trained_models/cvtslt_model_dev_19.80.pt --use_seqAE VAE
# test 20.06  dev 19.80
