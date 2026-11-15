import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib.pyplot as plt

from PIL import Image

def class_labels_to_pixel_mask(mask):
    CLASS_TO_PIXEL = {0: 0, 1: 126, 2: 189, 3: 63, 4: 252}
    out = np.zeros_like(mask, dtype=np.uint8)
    for cls, val in CLASS_TO_PIXEL.items():
        out[mask == cls] = val
    return out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False)
            )

        self.pool = nn.MaxPool2d(2)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        up4 = self.up4(c5)
        att4 = self.att4(g=up4, x=c4)
        up4 = torch.cat((att4, up4), dim=1)
        up4 = self.upconv4(up4)

        up3 = self.up3(up4)
        att3 = self.att3(g=up3, x=c3)
        up3 = torch.cat((att3, up3), dim=1)
        up3 = self.upconv3(up3)

        up2 = self.up2(up3)
        att2 = self.att2(g=up2, x=c2)
        up2 = torch.cat((att2, up2), dim=1)
        up2 = self.upconv2(up2)

        up1 = self.up1(up2)
        att1 = self.att1(g=up1, x=c1)
        up1 = torch.cat((att1, up1), dim=1)
        up1 = self.upconv1(up1)

        return self.final(up1)


PIXEL_TO_CLASS = {
    0:   0,
    126: 1,  # Left Kidney
    189: 2,  # Right Kidney
    63:  3,  # Liver
    252: 4  # Spleen
}

def normalized_mask_to_class_labels(mask):
    """
    Convert normalized float mask (0–1) to integer class labels (0–3)
    using known pixel-to-class mappings.
    
    Args:
        mask (np.ndarray): (H, W) float32 mask with values like 0.247, 0.4941

    Returns:
        np.ndarray: (H, W) uint8 mask with class labels (0, 1, 2, 3)
    """
    class_mask = np.zeros_like(mask, dtype=np.uint8)
    for pixel_val, class_id in PIXEL_TO_CLASS.items():
        normalized_val = pixel_val / 255.0
        class_mask[np.isclose(mask, normalized_val, atol=1e-3)] = class_id
    return class_mask


# Dataset loader
class NPYDataset(Dataset):
    def __init__(self, folders):
        self.image_mask_pairs = []
        for folder in folders:
            image = glob(os.path.join(folder, '*image.npy'))[0]
            mask = glob(os.path.join(folder, '*mask.npy'))[0]
            self.image_mask_pairs.append((image, mask))

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image and mask
        image = np.load(image_path).astype(np.float32)  # Already normalized
        mask = np.load(mask_path).astype(np.float32)    # Float values like 0.247, 0.4941

        # Convert float mask → class label map
        mask = normalized_mask_to_class_labels(mask)     # Now shape (H, W), uint8

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask, dtype=torch.long)                    # (H, W)

        return image, mask
    
def compute_class_iou(pred_mask, true_mask, num_classes=5, eps=1e-6):
    """
    Compute mean IoU for multi-class predictions using integer class labels.
    pred_mask: (B, H, W) - predicted class indices
    true_mask: (B, H, W) - ground truth class indices
    """
    batch_size = pred_mask.size(0)
    ious = []

    for b in range(batch_size):
        pred = pred_mask[b].cpu().numpy()
        true = true_mask[b].cpu().numpy()
        iou_total = 0
        class_count = 0

        for cls in range(1,num_classes):
            pred_cls = (pred == cls).astype(np.uint8)
            true_cls = (true == cls).astype(np.uint8)

            intersection = np.logical_and(pred_cls, true_cls).sum()
            union = np.logical_or(pred_cls, true_cls).sum()
            

            iou_total += (intersection + eps) / (union + eps)
            class_count += 1

        if class_count > 0:
            ious.append(iou_total / class_count)

    return np.mean(ious) if ious else 0.0

# Training

def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)              # shape: (B, 1, H, W)
        masks = masks.to(device)                # shape: (B, H, W), dtype: long

        optimizer.zero_grad()
        outputs = model(images)                 # shape: (B, C, H, W), raw logits


        loss = loss_fn(outputs, masks)          # CrossEntropyLoss expects (B, C, H, W) and (B, H, W)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



# Evaluation

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     total_iou = 0.0
#     count = 0

#     with torch.no_grad():
#         for images, masks in dataloader:
#             images, masks = images.to(device), masks.to(device)
#             outputs = model(images)
#             iou = compute_iou(outputs, masks)
#             total_iou += iou
#             count += 1

#     mean_iou = total_iou / count
#     print(f"Mean IoU: {mean_iou:.4f}")
#     return mean_iou


import os

def evaluate_model(model, dataloader, device, visualize=True, session_id=None, save_dir=None):
    model.eval()
    total_iou = 0.0
    count = 0

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            if session_id==1:
                iou = compute_class_iou(preds, masks, num_classes=3)
                total_iou += iou
                count += 1
            if session_id==2:
                iou = compute_class_iou(preds, masks, num_classes=4)
                total_iou += iou
                count += 1
            if session_id==3:
                iou = compute_class_iou(preds, masks, num_classes=5)
                total_iou += iou
                count += 1

            if visualize and idx < 5:
                input_img = images[0, 0].cpu().numpy()
                pred_mask = preds[0].cpu().numpy()
                true_mask = masks[0].cpu().numpy()

                pred_mask_vis = class_labels_to_pixel_mask(pred_mask)
                true_mask_vis = class_labels_to_pixel_mask(true_mask)

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(input_img, cmap='gray')
                axs[0].set_title('Input Image')

                axs[1].imshow(pred_mask_vis, cmap='spring', vmin=0, vmax=255)
                axs[1].set_title('Predicted Mask')

                axs[2].imshow(true_mask_vis, cmap='spring', vmin=0, vmax=255)
                axs[2].set_title('Ground Truth')

                for ax in axs:
                    ax.axis('off')
                plt.suptitle(f'Session {session_id} - Sample {idx + 1}')
                plt.tight_layout()

                if save_dir:
                    filename = f"session{session_id}_sample{idx+1}.png"
                    plt.savefig(os.path.join(save_dir, filename))
                else:
                    plt.show()

                plt.close()

    mean_iou = total_iou / count if count > 0 else 0.0
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou



# Setup session

def run_session(train_dirs, val_dir, session_id, model, device):
    print(f"\n--- Session {session_id} ---")

    random.seed(42)
    torch.manual_seed(42)

    # Randomly select up to 10 folders from each training directory
    train_folders = []
    for d in train_dirs:
        candidates = [os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
        train_folders.extend(random.sample(candidates, min(10, len(candidates))))

    val_folders = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))]

    train_loader = DataLoader(NPYDataset(train_folders), batch_size=2, shuffle=True)
    initial_images = []
    initial_masks = []

    raw_dataset = NPYDataset(train_folders)
    for img, msk in raw_dataset:
        initial_images.append(img)   # shape: (1, H, W)
        initial_masks.append(msk)    # shape: (H, W)

    # === Generate new data by recursive combination ===
    def combine_masks(mask1, mask2):
        combined = mask1.clone()
        m1 = mask1 != 0
        m2 = mask2 != 0
        combined[~m1 & m2] = mask2[~m1 & m2]
        both = m1 & m2
        combined[both] = torch.max(mask1[both], mask2[both])
        return combined

    def combine_images_with_masks(img1, img2, msk1, msk2):
        combined = img1.clone()
        m1 = msk1 != 0
        m2 = msk2 != 0
        combined[0][~m1 & m2] = img2[0][~m1 & m2]
        overlap = m1 & m2
        combined[0][overlap] = (img1[0][overlap] + img2[0][overlap]) / 2
        return combined

    def generate_combinations(images, masks, steps=2):
        all_images = images[:]
        all_masks = masks[:]
        current_images = images
        current_masks = masks

        for _ in range(steps):
            new_images, new_masks = [], []
            n = len(current_images)
            for i in range(n):
                for j in range(i+1, n):
                    img1, img2 = current_images[i], current_images[j]
                    msk1, msk2 = current_masks[i], current_masks[j]
                    new_img = combine_images_with_masks(img1, img2, msk1, msk2)
                    new_msk = combine_masks(msk1, msk2)
                    new_images.append(new_img)
                    new_masks.append(new_msk)

            if not new_images:
                break
            all_images.extend(new_images)
            all_masks.extend(new_masks)
            current_images = new_images
            current_masks = new_masks

        return all_images, all_masks

    # === Generate recursively expanded dataset ===
    expanded_images, expanded_masks = generate_combinations(initial_images, initial_masks, steps=2)  # Change steps as needed

    class CombinedDataset(Dataset):
        def __init__(self, images, masks):
            self.images = images
            self.masks = masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.masks[idx]

    # === Use combined dataset for training ===
    train_loader = DataLoader(CombinedDataset(expanded_images, expanded_masks), batch_size=2, shuffle=True)
    val_loader = DataLoader(NPYDataset(val_folders), batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train_model(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    evaluate_model(model, val_loader, device, visualize=True, session_id=session_id)


def run_session1(train_dirs, val_dir, session_id, model, device):
    print(f"\n--- Session {session_id} ---")

    random.seed(42)
    torch.manual_seed(42)

    # Randomly select 10 folders from each training dir
    train_folders = []
    for d in train_dirs:
        candidates = [os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
        train_folders.extend(random.sample(candidates, min(5, len(candidates))))

    val_folders = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))]

    train_loader = DataLoader(NPYDataset(train_folders), batch_size=2, shuffle=True)
    val_loader = DataLoader(NPYDataset(val_folders), batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):  # fewer epochs for example
        loss = train_model(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    evaluate_model(model, val_loader, device, session_id=session_id)

# Driver code

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=1, out_channels=3).to(device)

    # Specify your paths here
    left_kidney_train_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Leftkidney\Train1\1"
    right_kidney_train_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Rightkidney\Train1\1"
    kidney_val_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Val\Leftrightkidney"

    liver_train_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Liver\Train1\1"
    liver_val_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Val\Liver"

    spleen_train_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Spleen\Train1\1"
    spleen_val_dir = r"C:\Users\Ayantik\Desktop\ISI\Self Supervised and Active Learning\myfirstproject\Val\Spleen"

    run_session([left_kidney_train_dir, right_kidney_train_dir], kidney_val_dir, session_id=1, model=model, device=device)
    model = AttentionUNet(in_channels=1, out_channels=4).to(device)
    run_session1([liver_train_dir], liver_val_dir, session_id=2, model=model, device=device)
    model = AttentionUNet(in_channels=1, out_channels=5).to(device)
    run_session1([spleen_train_dir], spleen_val_dir, session_id=3, model=model, device=device)
    

if __name__ == "__main__":
    main()

