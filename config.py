import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    
    #load, input, save configurations:
    parser.add_argument('--out',help='output folder for checkpoint',default='./log/lstm_gcn/')
    parser.add_argument('--gap_save',help='gap between save model',default=50)
    parser.add_argument('--data',help='the path to dataset',default="./dataset/dance_music_paired.json")
    parser.add_argument('--pretrain_GCN',help='the pretrain GCN',default='./pretrain_model/GCN.pth')

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=800, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--lr_g', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--gap',help='train n iter if D while train 1 iter of G',default=1)
    parser.add_argument('--lr_d_frame', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--lr_d_seq', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=200)

    parser.add_argument('--encoder', type=str, help='gru, lstm, or tcn', default='gru')
    parser.add_argument('--resume', action='store_true', help='load weights and continue training')
    parser.add_argument('--gcn', action='store_true', help='use perceptual loss')
    return parser
 