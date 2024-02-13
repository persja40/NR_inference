import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.datasets import VOCDetection, Kitti
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# import openvino as ov
import pickle

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(batch=True):
    transforms = []
    if batch: # online pred is single image, while batch pred is multi image and requires same size
        transforms.append(T.Resize((512, 512)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def profile_FasterRCNN_ResNet50_FPN_V2(data_loader):
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.eval()
    # HERE OpenVino or other inference optimization framework can be applied, but this model is not supported and requires manual work
    # core = ov.Core()
    # ov_model = ov.convert_model(model)
    # compiled_model = core.compile_model(ov_model, 'CPU')

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True, profile_memory=True) as prof:
        with torch.no_grad():
            i = 0
            for images, _ in data_loader:
                if i>=1:
                    break # this break is to just get fast results on a single batch
                i += 1
                with record_function("model_inference"):
                    model(images) # results are not needed, it is inference test
    return prof

def save_profile(path, profile):
    results = []
    for event in profile.key_averages():
        results.append({
            "name": event.key,
            "cpu_time": event.cpu_time_total,
            "self_cpu_time": event.self_cpu_time_total,
            "cpu_memory_usage": event.cpu_memory_usage
        })
    with open(path, "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    dataset_val = VOCDetection('voc_root', year='2012', image_set='val', download=True, transform=get_transform())
    data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    prof = profile_FasterRCNN_ResNet50_FPN_V2(data_loader_val)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    save_profile("VOCDetection_FasterRCNN_ResNet50_FPN_V2.pkl", prof)

    dataset_val = Kitti('kitti_root', train=False, download=True, transform=get_transform())
    data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    prof = profile_FasterRCNN_ResNet50_FPN_V2(data_loader_val)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    save_profile("Kitti_FasterRCNN_ResNet50_FPN_V2.pkl", prof)
