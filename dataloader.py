import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import transforms
from transformers import AutoTokenizer, CLIPImageProcessor, AutoProcessor
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(SCRIPT_DIR, "weibo_dataset")
TRAIN_RUMOR_FILE = os.path.join(BASE_DIR, "train_rumor.jsonl")
TRAIN_NONRUMOR_FILE = os.path.join(BASE_DIR, "train_nonrumor.jsonl")
TEST_RUMOR_FILE = os.path.join(BASE_DIR, "test_rumor.jsonl")
TEST_NONRUMOR_FILE = os.path.join(BASE_DIR, "test_nonrumor.jsonl")

RUMOR_IMAGE_DIR = os.path.join(BASE_DIR, "rumor_images")
NONRUMOR_IMAGE_DIR = os.path.join(BASE_DIR, "nonrumor_images")





class WeiboRumorDataset(Dataset):
    def __init__(self, jsonl_files, image_base_path, label):
        self.data = []
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            './Qwen/Qwen3-VL-2B-Instruct', trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            './Qwen/Qwen3-VL-2B-Instruct', trust_remote_code=True
        )
        self.image_base_path = image_base_path

        TARGET_SIZE = 448
        self.transform_q = transforms.Compose([
            transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * TARGET_SIZE)+1, sigma=(0.1, 2.0)),
        ])

        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        image_filename = os.path.basename(item["image"])
                        image_path = os.path.join(self.image_base_path, image_filename)
                        if os.path.exists(image_path):
                            self.data.append({
                                "image_path": image_path,
                                "text": item["text"],
                                "label": label
                            })
                    except Exception as e:
                        print(f"Error loading line. Error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        text = item["text"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_q = self.transform_q(image)
        except Exception as e:
            print(f"[CORRUPTED FILE] {image_path}: {e}")
            return None


        text_inputs = self.qwen_tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=128
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}


        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "image_q": image_q,
            "labels": torch.tensor(item["label"], dtype=torch.long),
            #"pixel_values": pixel_values,
            #"image_grid_thw": image_grid_thw,
        }

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if isinstance(values[0], Image.Image):
            collated[key] = values
        else:
            collated[key] = torch.stack(values, dim=0)

    return collated




weibo_train_dataset_rumor = WeiboRumorDataset(
    [TRAIN_RUMOR_FILE], RUMOR_IMAGE_DIR, 1
)
weibo_train_dataset_nonrumor = WeiboRumorDataset(
    [TRAIN_NONRUMOR_FILE], NONRUMOR_IMAGE_DIR, 0
)
#twitter_train_dataset = TwitterDataset([TWITTER_TRAIN_FILE], TWITTER_TRAIN_DATA_DIR, bert_tokenizer,clip_image_processor)
full_train_dataset = ConcatDataset([weibo_train_dataset_rumor, weibo_train_dataset_nonrumor])

train_dataloader = DataLoader(
    full_train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=16,
    drop_last=True
)

test_dataset_rumor = WeiboRumorDataset(
    jsonl_files=[TEST_RUMOR_FILE],
    image_base_path=RUMOR_IMAGE_DIR,
    label=1
)

test_dataset_nonrumor = WeiboRumorDataset(
    jsonl_files=[TEST_NONRUMOR_FILE],
    image_base_path=NONRUMOR_IMAGE_DIR,
    label=0
)

#twitter_test_dataset = TwitterDataset([TWITTER_TEST_FILE], TWITTER_TEST_DATA_DIR, bert_tokenizer, clip_image_processor)

full_test_dataset = ConcatDataset([test_dataset_rumor, test_dataset_nonrumor])

test_dataloader = DataLoader(
    full_test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=16
)

if __name__ == '__main__':

    print("FINAL DATASET SUMMARY (Files Exist)")
    print(f"Weibo Rumor Train : {len(weibo_train_dataset_rumor)}")
    print(f"Weibo Non-Rumor Train : {len(weibo_train_dataset_nonrumor)}")
    print(f"Total Train Dataset : {len(full_train_dataset)}")
    print(f"Weibo Rumor Test : {len(test_dataset_rumor)}")
    print(f"Weibo Non-Rumor Test : {len(test_dataset_nonrumor)}")
    print(f"Total Test Dataset : {len(full_test_dataset)}")

