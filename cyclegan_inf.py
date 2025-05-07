import os
import torch
from torchvision import transforms
from PIL import Image
from ganv2.cyclegan_models import ResnetGenerator

def load_generator(checkpoint_path, device):
    """
    Load a ResnetGenerator from a .pth checkpoint.
    """
    net = ResnetGenerator(1, 1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net

def process_image(img_path, transform, generator, device):
    img = Image.open(img_path).convert('L')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = generator(tensor)
    out = out.squeeze(0).cpu()
    # unnormalize and clamp
    out = (out * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CycleGAN Inference')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save generated images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator checkpoint (.pth)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[120, 300], help='Resize dimensions (H, W)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_generator(args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize(tuple(args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('zed.jpg')):
            continue
        in_path = os.path.join(args.input_dir, fname)
        out_img = process_image(in_path, transform, generator, device)
        out_img.save(os.path.join(args.output_dir, fname))
    print('Inference complete.')