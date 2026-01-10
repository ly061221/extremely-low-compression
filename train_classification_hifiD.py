
import tqdm
import sys
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import os.path as osp
import torch.optim.lr_scheduler as LS
from backbone_cheng.msssim import *
from backbone_cheng.modules import *
from backbone_cheng.components import *
from backbone_cheng.enhance_dec import *
from GAN_Models.LPIPS import *
from GAN_hific.discriminator import *

import warnings
import argparse
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("-c", "--config",
                    default='config/classification_hifiD.yaml',
                    help="Path to config file")

given_configs, remaining = parser.parse_known_args(sys.argv[1:])
with open(given_configs.config) as file:
    yaml_data = yaml.safe_load(file)
    parser.set_defaults(**yaml_data)

args = parser.parse_args(remaining)


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
        self.decoder_enh = Decodercheng_featenh().to(self.device)
        self.resnet_decoder = ResNet_Transforms_module_192(192).to(self.device)
        self.Hyper_out = Hyper_Model(192).to(self.device)
        self.net_cls = ResNet50Modified(self.original_model_cls).to(self.device)
        self.hific_D = Discriminator((3, 256, 256), (192, 16, 16), 192)
        self.hific_D.to(self.device)
        # define GAN loss functions
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionLPIPS = LPIPSLoss(self.device)
        self.optimizer_dec_enh = optim.Adam(self.decoder_enh.parameters(), lr=args.learning_rate)
        self.optimizer_hific_D = optim.Adam(self.hific_D.parameters(), lr=args.learning_rate)
        self.scheduler_dec_enh = LS.MultiStepLR(self.optimizer_dec_enh, milestones=args.milestones, gamma=args.gamma)
        self.scheduler_hific_D = LS.MultiStepLR(self.optimizer_hific_D, milestones=args.milestones, gamma=args.gamma)

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
        self.min_lpips = 1
        self.nums_epoch = args.epochs
        self.clip_value = 1
        self.log.info(args)

    def _train_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder_enh.train()
        for param in self.decoder_enh.parameters():
            param.requires_grad = True
        for module in [
            self.decoder_enh.decoder_res1,
            self.decoder_enh.decoder_res2,
            self.decoder_enh.decoder_res3,
            self.decoder_enh.decoder_res4
        ]:
            module.eval()
            for param in module.parameters():
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

        self.hific_D.train()
        for param in self.hific_D.parameters():
            param.requires_grad = True

    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder_enh.eval()
        for param in self.decoder_enh.parameters():
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

        self.hific_D.eval()
        for param in self.hific_D.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.optimizer_dec_enh.zero_grad()
        self.optimizer_hific_D.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.decoder_enh.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.hific_D.parameters(), self.clip_value)
        self.optimizer_dec_enh.step()
        self.optimizer_hific_D.step()

    def update_lr(self):
        self.scheduler_dec_enh.step()
        self.scheduler_hific_D.step()

    def train_one_epoch(self):
        self.log.info('------------------------------Training------------------------------')
        self._train_init()
        G_loss_sum = []
        loss_G_lpips_sum = []
        loss_G_L2_sum = []
        loss_GAN_G_all_sum = []
        D_loss_sum = []

        tqdm_emu = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False)
        for batch, (d, l) in tqdm_emu:
            d = d.to(self.device)
            self._solver_init()
            enc_out = self.encoder(d)
            hyper_out = torch.round(enc_out)
            re_img = self.decoder_enh(hyper_out)
            latents = hyper_out

            ### D_GAN train ###
            # Fake Detection and Loss (re_img send into D_net), D loss
            D_fake_out, _ = self.hific_D.forward(re_img.detach(), latents.detach())
            D_real_out, _ = self.hific_D.forward(d.detach(), latents.detach())
            D_loss = GAN_D_loss(D_real_out, D_fake_out)
            # GAN loss (Fake Passability Loss)
            G_out, _ = self.hific_D.forward(re_img, latents)
            G_loss = GAN_G_loss(G_out)
            # lpips loss
            loss_G_lpips = (self.criterionLPIPS(re_img, d))
            # l2 loss between x and x'
            loss_G_L2 = self.criterionL2(re_img, d)
            loss_GAN_G_all = args.lambda_G * G_loss + args.lambda_L2 * loss_G_L2 + args.lambda_LPIPS * loss_G_lpips

            G_loss_sum.append(G_loss.item())
            loss_G_lpips_sum.append(loss_G_lpips.item())
            loss_G_L2_sum.append(loss_G_L2.item())
            D_loss_sum.append(D_loss.item())
            loss_GAN_G_all_sum.append(loss_GAN_G_all.item())

            D_loss.backward()
            loss_GAN_G_all.backward()
            self._solver_update()

            if (batch + 1) % 50 == 0:
                self.log.info(
                    'Epoch[{}/{}]({}/{}):\t'.format(self.epoch + 1, self.nums_epoch, batch + 1,
                                                    len(self.train_dataloader)) +
                    'loss_GAN_G_all:{:.5f};\t D_loss:{:.5f};\t'
                    'G_loss:{:.5f};\t loss_G_L2:{:.5f};\t loss_G_lpips:{:.5f}.'
                    .format(loss_GAN_G_all.item(), D_loss.item(),
                            G_loss.item(), loss_G_L2.item(), loss_G_lpips.item()))
        self.log.info(
            'lossG_all_mean:{:.5f};\t D_loss_mean:{:.5f};\t'
            'G_loss_mean:{:.5f};\t lossL2_mean:{:.5f};\t losslpips_mean:{:.5f}.'
                .format(np.mean(loss_GAN_G_all_sum), np.mean(D_loss_sum),
                        np.mean(G_loss_sum), np.mean(loss_G_L2_sum), np.mean(loss_G_lpips_sum)))
        self.epoch += 1

    def val_epoch(self):
        self.log.info('-----------------------------Evaluation-----------------------------')
        self._eval_init()
        bpp_loss = []
        psnr_item = []
        lpips_all = []

        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(self.val_dataloader))
            for i, (d, l) in tqdm_meter:
                d = d.to(self.device)
                enc_out = self.encoder(d)
                hyper_out = torch.round(enc_out)
                re_img = self.decoder_enh(hyper_out)

                # reconstruction_loss ：mse
                mse_img = mse(d, re_img)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
                # LPIPS
                lpips = lpips_distance(d, re_img, self.device)
                psnr_item.append(psnr)
                lpips_all.append(lpips)
        self.log.info(
            'R:{:.5f};\t psnr:{:.5f};\t LPIPS:{:.5f}.'
                .format(np.mean(bpp_loss), np.mean(psnr_item), np.mean(lpips_all)))
        if np.mean(lpips_all) < self.min_lpips:
            self.save_checkpoint('best.pth')
            self.min_lpips = np.mean(lpips_all)
        self.log.info('min_lpips: {:.5f}.'.format(self.min_lpips))

    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        ckp_path = osp.join(self.model_dir, model_name)
        self.log.info('Save checkpoint: %s' % ckp_path)
        obj = {
            'encoder': self.encoder.state_dict(),
            'decoder_enh': self.decoder_enh.state_dict(),
            'Hyper_out': self.Hyper_out.state_dict(),
            'resnet_decoder': self.resnet_decoder.state_dict(),
            'net_cls': self.net_cls.state_dict(),
            'hific_D': self.hific_D.state_dict(),
            'optimizer_dec_enh': self.optimizer_dec_enh.state_dict(),
            'optimizer_hific_D': self.optimizer_hific_D.state_dict(),
            'scheduler_dec_enh': self.scheduler_dec_enh.state_dict(),
            'scheduler_hific_D': self.scheduler_hific_D.state_dict(),
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
        self.decoder_enh.load_state_dict(obj['decoder_enh'])
        self.Hyper_out.load_state_dict(obj['Hyper_out'])
        self.resnet_decoder.load_state_dict(obj['resnet_decoder'])
        self.net_cls.load_state_dict(obj['net_cls'])
        self.hific_D.load_state_dict(obj['hific_D'])
        self.optimizer_dec_enh.load_state_dict(obj['optimizer_dec_enh'])
        self.optimizer_hific_D.load_state_dict(obj['optimizer_hific_D'])
        self.scheduler_dec_enh.load_state_dict(obj['scheduler_dec_enh'])
        self.scheduler_hific_D.load_state_dict(obj['scheduler_hific_D'])

    def load_checkpoint_decoderenh(self, args):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = args.checkpoint_enhdec_path
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.decoder_enh.load_state_dict(obj['decoder_enh'])
        self.Hyper_out.load_state_dict(obj['Hyper_out'])
        self.resnet_decoder.load_state_dict(obj['resnet_decoder'])
        self.net_cls.load_state_dict(obj['net_cls'])


def main():
    logger, model_dir = create_logger('logs', 'model', 'tbs', args.process, 'train')
    logger.info(args)
    trainer = train_compressai_cls(args, logger, model_dir)
    trainer.load_checkpoint_decoderenh(args)
    trainer.load_checkpoint(args)

    # trainer.val_epoch()
    for epoch in range(args.epochs):
        trainer.train_one_epoch()
        trainer.save_checkpoint('newest.pth')
        if epoch == args.epochs:
            trainer.save_checkpoint('final.pth')
        if epoch % 1 == 0:
            trainer.val_epoch()
        trainer.update_lr()


if __name__ == "__main__":
    main()


