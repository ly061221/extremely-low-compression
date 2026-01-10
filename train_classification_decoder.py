import tqdm
import sys
import torchvision
from torch.utils.data import DataLoader
import os.path as osp
from backbone_cheng.msssim import *
from backbone_cheng.modules import *
from backbone_cheng.components import *
from backbone_cheng.enhance_dec import *
import backbone_cheng.dataset as dataset
from GAN_Models.D_models import *
from GAN_Models.pytorch_msssim import *
from GAN_Models.LPIPS import *

import warnings
import argparse
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("-c", "--config",
                    default='config/decoder.yaml',
                    help="Path to config file")

given_configs, remaining = parser.parse_known_args(sys.argv[1:])
with open(given_configs.config) as file:
    yaml_data = yaml.safe_load(file)
    parser.set_defaults(**yaml_data)

args = parser.parse_args(remaining)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_logger(logname, modelname, boardname, cfg_name, phase='trainer'):
    root_log_dir = Path(logname)
    # set up logger
    if not root_log_dir.exists():
        print('=> creating {}'.format(root_log_dir))
        root_log_dir.mkdir()

    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_log_dir = root_log_dir / cfg_name

    print('=> creating {}'.format(final_log_dir))
    final_log_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y%m%d%H%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path(phase) / Path(modelname) / cfg_name / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(phase) / Path(boardname) / cfg_name / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_dir))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return logger, str(model_dir)


class train_compressai_cls():
    def __init__(self, args, logger, model_dir):
        super(train_compressai_cls, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_model_cls = resnet50(pretrained=False)
        self.original_model_cls.fc = nn.Linear(self.original_model_cls.fc.in_features, 102)
        self.encoder = Encodercheng().to(self.device)
        self.decoder_cheng = Decodercheng().to(self.device)
        self.decodercheng_enh = Decodercheng_featenh().to(self.device)
        self.resnet_decoder = ResNet_Transforms_module_192(192).to(self.device)  # 特征变换网络
        self.Hyper_out = Hyper_Model(192).to(self.device)  # 下游任务特征
        self.net_cls = ResNet50Modified(self.original_model_cls).to(self.device)  # 下游任务网络
        self.optimizer_deccheng_enh = optim.Adam(self.decodercheng_enh.parameters(), lr=args.learning_rate)

        cls_transforms_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        cls_transforms_val = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
        train_dataset = torchvision.datasets.ImageFolder(args.train_path, transform=cls_transforms_train)
        val_dataset = torchvision.datasets.ImageFolder(args.val_path, transform=cls_transforms_val)

        self.val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=(self.device == "cuda"))
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                           shuffle=True, pin_memory=(self.device == "cuda"))

        self.log = logger
        self.model_dir = model_dir
        self.epoch = 0
        self.min_loss = 100
        self.nums_epoch = args.epochs
        self.clip_value = 1
        self.log.info(args)

    def _train_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decodercheng_enh.train()
        for param in self.decodercheng_enh.parameters():
            param.requires_grad = True

        self.resnet_decoder.eval()
        for param in self.resnet_decoder.parameters():
            param.requires_grad = False

        self.Hyper_out.eval()
        for param in self.Hyper_out.parameters():
            param.requires_grad = False

        self.net_cls.eval()
        for param in self.net_cls.parameters():
            param.requires_grad = False

    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decodercheng_enh.eval()
        for param in self.decodercheng_enh.parameters():
            param.requires_grad = False

        self.resnet_decoder.eval()
        for param in self.resnet_decoder.parameters():
            param.requires_grad = False

        self.Hyper_out.eval()
        for param in self.Hyper_out.parameters():
            param.requires_grad = False

        self.net_cls.eval()
        for param in self.net_cls.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.optimizer_deccheng_enh.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.decodercheng_enh.parameters(), self.clip_value)
        self.optimizer_deccheng_enh.step()

    def train_no_discriminate(self):
        self.log.info('------------------------------Training------------------------------')
        self._train_init()

        tqdm_emu = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False)
        for batch, (d, l) in tqdm_emu:
            d = d.to(self.device)
            self._solver_init()
            enc_out = self.encoder(d)
            hyper_out = torch.round(enc_out).to(self.device)
            re_img = self.decodercheng_enh(hyper_out)

            # reconstruction_loss ：mse
            mse_img = mse(d, re_img)
            psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
            # rate_loss ：rate
            # R = rate_loss(hyper_out, d)
            total_loss = args.lmbda_mse * 255 ** 2 * mse_img
            total_loss.backward()
            self._solver_update()  # 梯度更新

            if (batch + 1) % 50 == 0:
                self.log.info(
                    'Epoch[{}/{}]({}/{}):\t'.format(self.epoch + 1, self.nums_epoch, batch + 1,
                                                    len(self.train_dataloader)) +
                    'loss:{:.7f};\t mse_loss:{:.7f};\t psnr:{:.5f}.'
                    .format(total_loss, mse_img, psnr))

        self.epoch += 1

    def test_no_discriminate(self):
        self.log.info('-----------------------------Evaluation-----------------------------')
        self._eval_init()

        bpp_loss = []
        mse_loss = []
        psnr_item = []
        lpips_all = []
        totalloss = []

        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(self.val_dataloader))
            for i, (d, l) in tqdm_meter:
                d = d.to(self.device)
                enc_out = self.encoder(d)
                hyper_out = torch.round(enc_out).to(self.device)
                re_img = self.decodercheng_enh(hyper_out)

                # reconstruction_loss ：mse
                mse_img = mse(d, re_img)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
                # LPIPS
                lpips = lpips_distance(d, re_img, self.device)
                total_loss = args.lmbda_mse * 255 ** 2 * mse_img

                mse_loss.append(mse_img.item())
                psnr_item.append(psnr)
                lpips_all.append(lpips)
                totalloss.append(total_loss.item())

        self.log.info(
            'R:{:.5f};\t psnr:{:.5f};\t'
            'LPIPS:{:.5f};\t loss:{:.5f};\t mse_loss:{:.5f}.'
                .format(np.mean(bpp_loss), np.mean(psnr_item),
                        np.mean(lpips_all), np.mean(totalloss), np.mean(mse_loss)))
        if np.mean(totalloss) < self.min_loss:
            self.save_checkpoint('best_loss.pth')
            self.min_loss = np.mean(totalloss)
        self.log.info('min_loss: {:.5f};\t Its BPP: {:.5f}.'.format(self.min_loss, np.mean(bpp_loss)))

    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        ckp_path = osp.join(self.model_dir, model_name)
        self.log.info('Save checkpoint: %s' % ckp_path)
        obj = {
            'encoder': self.encoder.state_dict(),
            'decoder_enh': self.decodercheng_enh.state_dict(),
            'Hyper_out': self.Hyper_out.state_dict(),
            'resnet_decoder': self.resnet_decoder.state_dict(),
            'net_cls': self.net_cls.state_dict(),
            'optimizer_dec_enh': self.optimizer_deccheng_enh.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.log.info('Save the trained model successfully.')

    def load_checkpoint(self, args):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = args.checkpoint_path
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.decodercheng_enh.load_state_dict(obj['decoder_enh'])
        self.Hyper_out.load_state_dict(obj['Hyper_out'])
        self.resnet_decoder.load_state_dict(obj['resnet_decoder'])
        self.net_cls.load_state_dict(obj['net_cls'])
        self.optimizer_deccheng_enh.load_state_dict(obj['optimizer_dec_enh'])

    def load_checkpoint_cls_RB(self, args):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = args.checkpoint_clsRB_path
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.decoder_cheng.load_state_dict(obj['decoder'])
        self.Hyper_out.load_state_dict(obj['Hyper_out'])
        self.resnet_decoder.load_state_dict(obj['resnet_decoder'])
        self.net_cls.load_state_dict(obj['net_cls'])

        old_state_dict = self.decoder_cheng.state_dict()
        new_state_dict = self.decodercheng_enh.state_dict()
        for old_key, value in old_state_dict.items():
            if old_key.startswith('decoder.'):
                if old_key.startswith('decoder.0'):
                    key_suffix = old_key.split('decoder.0', 1)[-1]
                    new_key = 'decoder_res1.0' + key_suffix
                elif old_key.startswith('decoder.1'):
                    key_suffix = old_key.split('decoder.1', 1)[-1]
                    new_key = 'decoder_res1.1' + key_suffix
                elif old_key.startswith('decoder.2'):
                    key_suffix = old_key.split('decoder.2', 1)[-1]
                    new_key = 'decoder_res2.0' + key_suffix
                elif old_key.startswith('decoder.3'):
                    key_suffix = old_key.split('decoder.3', 1)[-1]
                    new_key = 'decoder_res2.1' + key_suffix
                elif old_key.startswith('decoder.4'):
                    key_suffix = old_key.split('decoder.4', 1)[-1]
                    new_key = 'decoder_res3.0' + key_suffix
                elif old_key.startswith('decoder.5'):
                    key_suffix = old_key.split('decoder.5', 1)[-1]
                    new_key = 'decoder_res3.1' + key_suffix
                elif old_key.startswith('decoder.6'):
                    key_suffix = old_key.split('decoder.6', 1)[-1]
                    new_key = 'decoder_res3.2' + key_suffix
                elif old_key.startswith('decoder.7'):
                    key_suffix = old_key.split('decoder.7', 1)[-1]
                    new_key = 'decoder_res4.0' + key_suffix
                elif old_key.startswith('decoder.8'):
                    key_suffix = old_key.split('decoder.8', 1)[-1]
                    new_key = 'decoder_res4.1' + key_suffix
                elif old_key.startswith('decoder.9'):
                    key_suffix = old_key.split('decoder.9', 1)[-1]
                    new_key = 'decoder_outimg' + key_suffix
                else:
                    print("NONE!!!")
                new_state_dict[new_key] = value

        self.decodercheng_enh.load_state_dict(new_state_dict)


def main():
    logger, model_dir = create_logger('logs', 'model', 'tbs', args.process, 'train')
    logger.info(args)
    trainer = train_compressai_cls(args, logger, model_dir)
    trainer.load_checkpoint_cls_RB(args)
    trainer.load_checkpoint(args)
    trainer.test_no_discriminate()

    for epoch in range(args.epochs):
        trainer.train_no_discriminate()
        trainer.save_checkpoint('newest.pth')
        if epoch == args.epochs:
            trainer.save_checkpoint('final.pth')
        if epoch % 1 == 0:
            trainer.test_no_discriminate()


if __name__ == "__main__":
    main()
