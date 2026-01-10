import tqdm
import sys
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from backbone_cheng.msssim import *
from backbone_cheng.modules import *
from backbone_cheng.components import *
from backbone_cheng.enhance_dec import *
from GAN_Models.D_models import *
from GAN_Models.pytorch_msssim import *
from GAN_Models.LPIPS import *
import warnings
import argparse
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("-c", "--config",
                    default=r"config/eval.yaml",
                    help="Path to config file")
# parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str,
#                     help='Result dir name')

given_configs, remaining = parser.parse_known_args(sys.argv[1:])
with open(given_configs.config) as file:
    yaml_data = yaml.safe_load(file)
    parser.set_defaults(**yaml_data)

# parser.add_argument("-T", "--TEST", default=False, help='Testing')
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


def create_logger(logname, cfg_name, phase='trainer'):
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
    return logger


class train_compressai_cls():
    def __init__(self, args, logger):
        super(train_compressai_cls, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_model_cls = resnet50(pretrained=False)
        self.original_model_cls.fc = nn.Linear(self.original_model_cls.fc.in_features, 102)
        self.encoder = Encodercheng().to(self.device)
        self.decoder_enh = Decodercheng_featenh().to(self.device)
        self.resnet_decoder = ResNet_Transforms_module_192(192).to(self.device)
        self.Hyper_out = Hyper_Model(192).to(self.device)
        self.net_cls = ResNet50Modified(self.original_model_cls).to(self.device)

        cls_transforms_val = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
        val_dataset = torchvision.datasets.ImageFolder(args.val_path, transform=cls_transforms_val)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=(self.device == "cuda"))

        self.log = logger
        self.epoch = 0
        self.log.info(args)

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

    def val_epoch(self):
        self.log.info('-----------------------------Evaluation-----------------------------')
        self._eval_init()
        bpp_loss = []
        psnr_item = []
        top1_accuracy = []
        lpips_all = []

        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(self.val_dataloader))
            for i, (d, l) in tqdm_meter:
                d = d.to(self.device)
                l = l.to(self.device)
                enc_out = self.encoder(d)
                hyper_out = self.Hyper_out(enc_out)
                feat_trans = self.resnet_decoder(hyper_out["y_hat"])
                pred = self.net_cls(feat_trans)
                re_img = self.decoder_enh(hyper_out["y_hat"])

                # reconstruction_loss ：mse
                mse_img = mse(d, re_img)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse_img))
                # LPIPS
                lpips = lpips_distance(d, re_img, self.device)
                top1_acc, _ = accuracy(pred, l, topk=(1, 5))
                # rate_loss ：rate
                R = rate_loss(hyper_out, d)

                bpp_loss.append(R.item())
                psnr_item.append(psnr)
                lpips_all.append(lpips)
                top1_accuracy.append(top1_acc.item())

        self.log.info(
            'R:{:.5f};\t psnr:{:.5f};\t LPIPS:{:.5f};\t top1_acc:{:.5f}.'
                .format(np.mean(bpp_loss), np.mean(psnr_item), np.mean(lpips_all), np.mean(top1_accuracy)))

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


def main():
    logger = create_logger('logs', args.process, 'val')
    logger.info(args)
    trainer = train_compressai_cls(args, logger)
    trainer.load_checkpoint(args)
    trainer.val_epoch()


if __name__ == "__main__":
    main()
