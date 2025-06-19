import torch
from collections import OrderedDict
from src.models.backbone import FeatureExtractor
from external.DRCT.drct.models.drct_model import DRCTModel

from basicsr.losses import build_loss
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY



@MODEL_REGISTRY.register()
class DRCTModelFinal(DRCTModel):
    def __init__(self, opt):
        super(DRCTModelFinal, self).__init__(opt)

        train_opt = self.opt["train"]
        self.pixel_weight_loss = train_opt.get("pixel_loss_weight", 1)
        self.context_weight_loss = train_opt.get("context_loss_weight", 1)
        self.gradient_accumulation_steps = train_opt.get("gradient_accumulation_steps", 1)
        self.feat_extractor_path = self.opt.get('feat_extractor_path', None)
        if self.feat_extractor_path:
            self.feat_extractor = FeatureExtractor(self.feat_extractor_path).to(self.device)
            self.feat_extractor.eval()
            print(f'Feature extractor loaded from {self.feat_extractor_path}')


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.pix_loss = build_loss(train_opt['pixel_loss']).to(self.device)
        if train_opt.get("context_loss") is not None:
            self.ctx_loss = build_loss(train_opt['context_loss']).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()
    
    
    def optimize_parameters(self, current_iter):
        self.output = self.net_g(self.lq)

        if hasattr(self, "feat_extractor") and hasattr(self, "ctx_loss"):
            gt_feat = self.gt.repeat(1, 3, 1, 1)
            output_feat = self.output.repeat(1, 3, 1, 1)

            self.feat_y = self.feat_extractor(gt_feat)
            self.feat_pred = self.feat_extractor(output_feat)

        l_total = 0
        loss_dict = OrderedDict()
        l_pixel = self.pix_loss(self.output, self.gt)
        loss_dict['l_pixel'] = l_pixel
        if hasattr(self, "ctx_loss"):
            l_total += l_pixel * self.pixel_weight_loss
            loss_dict['l_pixel_wgt'] = l_pixel * self.pixel_weight_loss
        else:
            l_total += l_pixel
        
        if hasattr(self, "ctx_loss"):
            l_context = self.ctx_loss(self.feat_pred, self.feat_y)
            loss_dict['l_context'] = l_context
            loss_dict["l_context_wgt"] = l_context * self.context_weight_loss
            l_total += l_context * self.context_weight_loss

        l_total = l_total /  self.gradient_accumulation_steps
        l_total.backward()
        
        if current_iter % self.gradient_accumulation_steps == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

            if self.ema_decay > 0:
                self.model_ema(self.ema_decay)
            
        self.log_dict = self.reduce_loss_dict(loss_dict)
        