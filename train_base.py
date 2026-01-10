
import tqdm
import sys
import torchvision
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
import os.path as osp
import torch.optim.lr_scheduler as LS
from backbone_cheng.msssim import *
from backbone_cheng.modules import *
from backbone_cheng.components import *
from GAN_Models.pytorch_msssim import *
from GAN_Models.LPIPS import *
import warnings
import argparse
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("-c", "--config",
                    default='config/base_codec.yaml',
                    help="Path to config file")
# parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str,
#                     help='Result dir name')

given_configs, remaining = parser.parse_known_args(sys.argv[1:])
with open(given_configs.config) as file:
    yaml_data = yaml.safe_load(file)
    parser.set_defaults(**yaml_data)

# parser.add_argument("-T", "--TEST", default=False, help='Testing')
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

        self.encoder = Encodercheng().to(self.device)
        self.decoder = Decodercheng().to(self.device)
        self.Hyper_out = Hyper_Model(192).to(self.device)

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.learning_rate)
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.learning_rate)
        self.optimizer_Hyper_out = optim.Adam(self.Hyper_out.parameters(), lr=args.learning_rate)

        self.scheduler_enc = LS.MultiStepLR(self.optimizer_enc, milestones=args.milestones, gamma=args.gamma)
        self.scheduler_dec = LS.MultiStepLR(self.optimizer_dec, milestones=args.milestones, gamma=args.gamma)
        self.scheduler_Hyper_out = LS.MultiStepLR(self.optimizer_Hyper_out, milestones=args.milestones,
                                                  gamma=args.gamma)

        transforms_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transforms_val = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.ImageFolder(args.train_path, transform=transforms_train)
        val_dataset = torchvision.datasets.ImageFolder(args.val_path, transform=transforms_val)

        self.val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=(self.device == "cuda"))
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                           shuffle=True, pin_memory=(self.device == "cuda"))

        self.log = logger
        self.model_dir = model_dir
        self.epoch = 0
        self.min_loss = 100
        self.nums_epoch = args.epochs
        self.clip_value = 1
        self.log.info("lam_mse:{}".format(args.lmbda_mse))
        self.log.info(args)

    def _train_init(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True

        self.Hyper_out.train()
        for param in self.Hyper_out.parameters():
            param.requires_grad = True

    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.Hyper_out.eval()
        for param in self.Hyper_out.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.optimizer_dec.zero_grad()
        self.optimizer_enc.zero_grad()
        self.optimizer_Hyper_out.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.decoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.Hyper_out.parameters(), self.clip_value)

        self.optimizer_enc.step()
        self.optimizer_dec.step()
        self.optimizer_Hyper_out.step()

    def update_lr(self):
        self.scheduler_enc.step()
        self.scheduler_dec.step()
        self.scheduler_Hyper_out.step()

    def train_one_epoch(self):
        self.log.info('------------------------------Training------------------------------')
        self._train_init()

        tqdm_emu = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False)
        for batch, (d, l) in tqdm_emu:
            d = d.to(self.device)
            self._solver_init()
            enc_out = self.encoder(d)
            hyper_out = self.Hyper_out(enc_out)
            re_img = self.decoder(hyper_out["y_hat"])

            # reconstruction_loss ：mse
            mse_img = mse(d, re_img)
            psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
            # rate_loss ：rate
            R = rate_loss(hyper_out, d)
            total_loss = args.lmbda_mse * (255 ** 2 * mse_img) + R
            total_loss.backward()
            self._solver_update()

            if (batch + 1) % 50 == 0:
                self.log.info(
                    'Epoch[{}/{}]({}/{}):\t'.format(self.epoch + 1, self.nums_epoch, batch + 1,
                                                    len(self.train_dataloader)) +
                    'loss:{:.7f};\t'
                    'mse_loss:{:.7f};\t psnr:{:.5f};\t R:{:.5f}.'
                    .format(total_loss,
                            mse_img, psnr, R))

        self.epoch += 1

    def val_epoch(self):
        self.log.info('-----------------------------Evaluation-----------------------------')
        self._eval_init()

        bpp_loss = []
        mse_loss = []
        psnr_item = []
        totalloss = []
        lpips_all = []
        msssim_all = []

        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(self.val_dataloader))
            for i, (d, l) in tqdm_meter:
                d = d.to(self.device)
                enc_out = self.encoder(d)
                hyper_out = self.Hyper_out(enc_out)
                re_img = self.decoder(hyper_out["y_hat"])

                # reconstruction_loss ：mse
                mse_img = mse(d, re_img)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
                # rate_loss ：rate
                R = rate_loss(hyper_out, d)
                # LPIPS
                lpips = lpips_distance(d, re_img, self.device)
                # MSSSIM
                ms_ssim_value = ms_ssim(d, re_img, data_range=1.0, size_average=True)
                total_loss = args.lmbda_mse * (255 ** 2 * mse_img) + R

                bpp_loss.append(R.item())
                mse_loss.append(mse_img.item())
                psnr_item.append(psnr)
                lpips_all.append(lpips)
                msssim_all.append(ms_ssim_value.item())
                totalloss.append(total_loss.item())

        self.log.info(
            'loss:{:.7f};\t'
            'mse_loss:{:.7f};\t R:{:.5f};\t psnr:{:.5f};\t'
            'LPIPS:{:.5f};\t msssim:{:.5f}.'
                .format(np.mean(totalloss),
                        np.mean(mse_loss), np.mean(bpp_loss), np.mean(psnr_item),
                        np.mean(lpips_all), np.mean(msssim_all)))

        if np.mean(totalloss) < self.min_loss:
            self.save_checkpoint('best_loss.pth')
            self.min_loss = np.mean(totalloss)
        self.log.info('min_loss: {:.7f};\t Its BPP: {:.5f}.'.format(self.min_loss, np.mean(bpp_loss)))

    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        ckp_path = osp.join(self.model_dir, model_name)
        self.log.info('Save checkpoint: %s' % ckp_path)
        obj = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'Hyper_out': self.Hyper_out.state_dict(),

            'optimizer_enc': self.optimizer_enc.state_dict(),
            'optimizer_dec': self.optimizer_dec.state_dict(),
            'optimizer_Hyper_out': self.optimizer_Hyper_out.state_dict(),

            'scheduler_enc': self.scheduler_enc.state_dict(),
            'scheduler_dec': self.scheduler_dec.state_dict(),
            'scheduler_Hyper_out': self.scheduler_Hyper_out.state_dict(),
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
        self.decoder.load_state_dict(obj['decoder'])
        self.Hyper_out.load_state_dict(obj['Hyper_out'])

        self.optimizer_enc.load_state_dict(obj['optimizer_enc'])
        self.optimizer_dec.load_state_dict(obj['optimizer_dec'])
        self.optimizer_Hyper_out.load_state_dict(obj['optimizer_Hyper_out'])

        self.scheduler_enc.load_state_dict(obj['scheduler_enc'])
        self.scheduler_dec.load_state_dict(obj['scheduler_dec'])
        self.scheduler_Hyper_out.load_state_dict(obj['scheduler_Hyper_out'])


def main():

    logger, model_dir = create_logger('logs', 'model', 'tbs', args.process, 'train')
    logger.info(args)
    trainer = train_compressai_cls(args, logger, model_dir)
    trainer.load_checkpoint(args)
    # trainer.val_epoch()

    for epoch in range(args.epochs):
        trainer.train_one_epoch()
        trainer.save_checkpoint('newest.pth')
        if epoch == args.epochs:
            trainer.save_checkpoint('final.pth')
        if epoch % 5 == 0:
            trainer.val_epoch()

        trainer.update_lr()


if __name__ == "__main__":
    main()

