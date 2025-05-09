

def collate_fn(batch):
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]

    heights = [img.shape[1] for img in images]  
    widths = [img.shape[2] for img in images]

    max_height = max(heights)
    max_width = max(widths)

    padded_images = []
    padded_masks = []

    for image, mask in zip(images, masks):
        _, h, w = image.shape
        pad_height = max_height - h
        pad_width = max_width - w
        padding = (0, 0, pad_width, pad_height)  # left, top, right, bottom

        padded_images.append(F.pad(image, padding, fill=0))
        padded_masks.append(F.pad(mask, padding, fill=0))

    return {
        'images': torch.stack(padded_images),  
        'masks': torch.stack(padded_masks)     
    }

class LineSegmentationDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, segs: np.ndarray, lines: np.ndarray):
        self.line_images = []
        self.line_char_segmentations = []

        for scroll, seg, line in zip(scrolls, segs, lines, strict=True):
            li, ls = synthetic.extract_lines_segs_cc(scroll, seg, line)
            self.line_images.extend(li)
            self.line_char_segmentations.extend(ls)

    def __len__(self):
        return len(self.line_images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.line_images[idx]).convert("L")  
        image = F.to_tensor(image)  

        mask = torch.tensor(self.line_char_segmentations[idx], dtype=torch.long)  

        return {
            'image': image,
            'mask': mask
        }