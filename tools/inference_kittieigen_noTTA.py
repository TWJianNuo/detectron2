#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import time
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    Kitti2cityscapesInstanceEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    DatasetEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    print_csv_format,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import log_every_n_seconds

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances
from cityscapesscripts.helpers.labels import labels

import cv2
from collections import deque
from contextlib import contextmanager
import datetime
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt

def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return Image.fromarray(tnp[:, :, 0:3])

def vls_ins(rgb, anno):
    rgbc = copy.deepcopy(rgb)
    r = rgbc[:, :, 0].astype(np.float)
    g = rgbc[:, :, 1].astype(np.float)
    b = rgbc[:, :, 2].astype(np.float)
    for i in np.unique(anno):
        if i > 0:
            rndc = np.random.randint(0, 255, 3).astype(np.float)
            selector = anno == i
            r[selector] = rndc[0] * 0.25 + r[selector] * 0.75
            g[selector] = rndc[1] * 0.25 + g[selector] * 0.75
            b[selector] = rndc[2] * 0.25 + b[selector] * 0.75
    rgbvls = np.stack([r, g, b], axis=2)
    rgbvls = np.clip(rgbvls, a_max=255, a_min=0).astype(np.uint8)
    return rgbvls

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.catconfbar = {0 : 0.9, 1 : 0.9, 2 : 0.9}
        self.minpixel = {0: 50, 1: 100, 2: 100}
        self.selfcontribbar = {0: 0.5, 1: 0.5, 2: 0.5}

    def pp_predictions_simple(self, inspred):
        selector = torch.ones_like(inspred.scores)
        if inspred.scores.shape[0] == 0:
            return inspred
        for k in range(inspred.scores.shape[0]):
            cat = inspred.pred_classes[k].item()
            conf = inspred.scores[k].item()
            numpixel = torch.sum(inspred.pred_masks).item()

            if conf < self.catconfbar[cat]:
                selector[k] = 0

            if numpixel < self.minpixel[cat]:
                selector[k] = 0

        pp_inspred = Instances(image_size=inspred.image_size)

        selector = selector == 1
        pp_inspred.scores = inspred.scores[selector]
        pp_inspred.pred_classes = inspred.pred_classes[selector]
        pp_inspred.pred_boxes = inspred.pred_boxes[selector]
        pp_inspred.pred_masks = inspred.pred_masks[selector]

        return pp_inspred

    def erase_srhink(self, inspred, mask, cat):
        catidx = list()
        catscore = list()
        for k in range(len(inspred)):
            if inspred.pred_classes[k].item() == cat:
                catidx.append(k)
                catscore.append(inspred.scores[k].item())

        if len(catidx) == 0:
            return inspred
        else:
            catidx = np.array(catidx)
            catscore = np.array(catscore)
            sortedidx = np.argsort(catscore)

            catidx = catidx[sortedidx]
            catscore = catscore[sortedidx]

            refmask = np.copy(mask)
            refmask = torch.from_numpy(refmask) == 1

            # tensor2disp(refmask.unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
            for k in range(catidx.shape[0]):
                if catscore[k] < self.catconfbar[cat]:
                    inspred = self.erase(inspred, catidx, catidx[k], cat)
                inspred.pred_masks[catidx[k]] = inspred.pred_masks[catidx[k]] * refmask
            return inspred

    def erase(self, inspred, catidx, selfidx, cat):
        mask_wos = torch.zeros_like(inspred.pred_masks[selfidx])

        for k in catidx:
            if k == selfidx:
                continue
            else:
                mask_wos += inspred.pred_masks[k]
        mask_ws = mask_wos + inspred.pred_masks[selfidx]
        solcontrib = (mask_wos == 0) * (mask_ws == 1)
        if torch.sum(solcontrib).float() / (torch.sum(inspred.pred_masks[selfidx]) + 1).float() < self.selfcontribbar[cat]:
            # erase
            inspred.pred_masks[selfidx] = inspred.pred_masks[selfidx] * 0
        return inspred

    def pp_predictions(self, inspred, carmask, pedmask, cyclistmask):
        selector = torch.ones_like(inspred.scores)
        if inspred.scores.shape[0] == 0:
            return inspred

        # Get indices sort by score
        inspred = self.erase_srhink(inspred, carmask, cat=2)
        inspred = self.erase_srhink(inspred, cyclistmask, cat=1)
        inspred = self.erase_srhink(inspred, pedmask, cat=0)

        for k in range(inspred.scores.shape[0]):
            cat = inspred.pred_classes[k].item()
            numpixel = torch.sum(inspred.pred_masks[k]).item()

            if numpixel < self.minpixel[cat]:
                selector[k] = 0

        pp_inspred = Instances(image_size=inspred.image_size)

        selector = selector == 1
        pp_inspred.scores = inspred.scores[selector]
        pp_inspred.pred_classes = inspred.pred_classes[selector]
        pp_inspred.pred_boxes = inspred.pred_boxes[selector]
        pp_inspred.pred_masks = inspred.pred_masks[selector]

        return pp_inspred

    def generate_instancemap(self, inspred, h, w):
        insmap = torch.zeros([h, w], dtype=torch.int32)
        semanmap = torch.zeros([h, w], dtype=torch.int32)
        if len(inspred) == 0:
            return insmap, semanmap
        else:
            scores = inspred.scores.numpy()
            idx = np.argsort(-scores)
            for num, k in enumerate(idx):
                insmap[inspred.pred_masks[k]] = num + 1
                semanmap[inspred.pred_masks[k]] = inspred.pred_classes[k].item() + 1
            return insmap, semanmap

    def run_on_image(self, image, predictions, carmask, pedmask, cyclistmask, entryname, args):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        foldname, imgname = entryname.split(' ')

        dirmapping = {'left': 'image_02', 'right': 'image_03'}
        date = foldname[0:10]
        seq = foldname[0:26]
        foldname = dirmapping[foldname.split('_')[-1]]

        exportfold_ins = os.path.join(args.exportroot, date, seq, 'insmap', foldname)
        exportfold_seman = os.path.join(args.exportroot, date, seq, 'semanmap', foldname)

        ins_path = os.path.join(exportfold_ins, imgname)
        seman_path = os.path.join(exportfold_seman, imgname)

        if os.path.exists(ins_path) and os.path.exists(seman_path):
            print("%s generated, skip" % ins_path)
            return

        os.makedirs(exportfold_ins, exist_ok=True)
        os.makedirs(exportfold_seman, exist_ok=True)

        instances = predictions["instances"].to(self.cpu_device)

        if carmask is None:
            pp_inspred = self.pp_predictions_simple(copy.deepcopy(instances))
        else:
            pp_inspred = self.pp_predictions(copy.deepcopy(instances), carmask, pedmask, cyclistmask)

        insmap, semanmap = self.generate_instancemap(pp_inspred, h=image.shape[0], w=image.shape[1])

        Image.fromarray(insmap.numpy().astype(np.uint8)).save(ins_path)
        Image.fromarray(semanmap.numpy().astype(np.uint8)).save(seman_path)

        if np.random.randint(0, args.vlsfreq) == 0:
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]

            vlsfold1 = os.path.join(args.vlsroot, date, seq, 'vls_final')
            vlsfold2 = os.path.join(args.vlsroot, date, seq, 'vls_initial')
            vlsfold3 = os.path.join(args.vlsroot, date, seq, 'vls_cleaned')

            vlsname1 = os.path.join(vlsfold1, imgname)
            vlsname2 = os.path.join(vlsfold2, imgname)
            vlsname3 = os.path.join(vlsfold3, imgname)

            os.makedirs(vlsfold1, exist_ok=True)
            os.makedirs(vlsfold2, exist_ok=True)
            os.makedirs(vlsfold3, exist_ok=True)

            Image.fromarray(vls_ins(rgb=image, anno=insmap.numpy())).save(vlsname1)

            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            Image.fromarray(vis_output.get_image()).save(vlsname2)

            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            vis_output = visualizer.draw_instance_predictions(predictions=pp_inspred)
            Image.fromarray(vis_output.get_image()).save(vlsname3)
        return


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def get_semantics(args, entryname, h, w):
    lrmapping = {'left': 'image_02', 'right': 'image_03'}

    foldname, imgname = entryname.split(' ')
    date = foldname[0:10]
    seq = foldname[0:26]

    carsemancat = [26, 27, 28, 29, 30, 31]
    pedsemancat = [24]
    cyclistsemancat = [25, 32, 33]

    carmask = None
    pedmask = None
    cyclistmask = None

    semanticspath = os.path.join(args.semanticsroot, date, seq, 'semantic_prediction', lrmapping[foldname[27::]], imgname)
    if os.path.exists(semanticspath):
        semantics = Image.open(semanticspath).resize([w, h], Image.NEAREST)
        semantics = np.array(semantics)

        carmask = np.zeros_like(semantics, dtype=np.float32)
        for c in carsemancat:
            carmask = carmask + (semantics == c).astype(np.float32)

        pedmask = np.zeros_like(semantics, dtype=np.float32)
        for c in pedsemancat:
            pedmask = pedmask + (semantics == c).astype(np.float32)

        cyclistmask = np.zeros_like(semantics, dtype=np.float32)
        for c in cyclistsemancat:
            cyclistmask = cyclistmask + (semantics == c).astype(np.float32)

    return carmask, pedmask, cyclistmask

def inference_on_dataset(model, data_loader, vlstool, args):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            carmask, pedmask, cyclistmask = get_semantics(args, inputs[0]['entryname'], inputs[0]['height'], inputs[0]['width'])
            vlstool.run_on_image(inputs[0]['orgimage'].cpu().permute([1, 2, 0]).numpy(), outputs[0], carmask, pedmask, cyclistmask, inputs[0]['entryname'], args)
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
            eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
            print("Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, total, seconds_per_img, str(eta)
                ))
    return


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def inference_without_TTA(cls, cfg, model, args):
        cls.inference(cfg, model, args)
        return

    @classmethod
    def inference(cls, cfg, model, args):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            vlstool = VisualizationDemo(cfg)
            inference_on_dataset(model, data_loader, vlstool, args)
        return


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # res = Trainer.test_with_TTA(cfg, model)
        Trainer.inference_without_TTA(cfg, model, args)
        return
    else:
        raise Exception("Only evaluation supported")

def build_inference_dataset(args, removeorg=True):
    # Remove original test split
    from shutil import copyfile, rmtree
    from tqdm import tqdm
    import glob

    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0016_sync',
        '2011_09_30/2011_09_30_drive_0018_sync',
        '2011_09_30/2011_09_30_drive_0027_sync'
    ]

    if removeorg:
        orgTlr = os.path.join(args.kitti2cityscaperoot, 'gtFine/test')
        orgTir = os.path.join(args.kitti2cityscaperoot, 'leftImg8bit/test')
        if os.path.exists(orgTlr) and os.path.isdir(orgTlr):
            print("Removing: %s" % orgTlr)
            rmtree(orgTlr)
        if os.path.exists(orgTir) and os.path.isdir(orgTir):
            print("Removing: %s" % orgTir)
            rmtree(orgTir)

    txts = ['test_files.txt', 'val_files.txt', 'train_files.txt']
    splitfolder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'split_eigen_full', )

    entries = list()
    for txtname in txts:
        splittxtadd = os.path.join(splitfolder, txtname)
        with open(splittxtadd, 'r') as f:
            tmpentries = f.readlines()
            for entry in tmpentries:
                seq, frmidx, dir = entry.split(' ')

                key = "{} {}".format(seq, frmidx.zfill(10))
                entries.append(key)

    entries = list(set(entries))
    srcs = list()
    dsts = list()

    for entry in entries:
        seq, frmidx = entry.split(' ')
        srcl = os.path.join(args.rawkittiroot, seq, 'image_02/data', "{}.png".format(frmidx))
        dstfoldl = os.path.join(args.kitti2cityscaperoot, 'leftImg8bit/test', '{}_left'.format(seq.split('/')[-1]))
        dstl = os.path.join(dstfoldl, "{}.png".format(frmidx))

        srcs.append(srcl)
        dsts.append(dstl)

        srcr = os.path.join(args.rawkittiroot, seq, 'image_03/data', "{}.png".format(frmidx))
        dstfoldr = os.path.join(args.kitti2cityscaperoot, 'leftImg8bit/test', '{}_right'.format(seq.split('/')[-1]))
        dstr = os.path.join(dstfoldr, "{}.png".format(frmidx))

        os.makedirs(dstfoldl, exist_ok=True)
        os.makedirs(dstfoldr, exist_ok=True)

        srcs.append(srcr)
        dsts.append(dstr)

    assert len(entries) * 2 == len(srcs)
    for odomseq in odomseqs:
        dstfoldl = os.path.join(args.kitti2cityscaperoot, 'leftImg8bit/test', '{}_left'.format(seq.split('/')[-1]))
        leftimgs = glob.glob(os.path.join(args.kittiodomroot, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)

            srcl = os.path.join(args.kittiodomroot, odomseq, 'image_02/data', imgname)
            dstl = os.path.join(dstfoldl, imgname)

            srcs.append(srcl)
            dsts.append(dstl)
        os.makedirs(dstfoldl, exist_ok=True)

    for k in tqdm(range(len(entries) * 2)):
        if os.path.exists(dsts[k]):
            continue
        else:
            copyfile(srcs[k], dsts[k])

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--rawkittiroot", type=str)
    parser.add_argument("--kitti2cityscaperoot", type=str)
    parser.add_argument("--kittiodomroot", type=str)
    parser.add_argument("--semanticsroot", type=str)
    parser.add_argument("--banremove", action='store_true')
    parser.add_argument("--exportroot", type=str)
    parser.add_argument("--vlsroot", type=str)
    parser.add_argument("--vlsfreq", type=int, default=100)
    args = parser.parse_args()

    build_inference_dataset(args, removeorg=not args.banremove)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
