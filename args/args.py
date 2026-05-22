import argparse

parser = argparse.ArgumentParser(description="Train Model on 5-class ImageNet subset")
parser.add_argument('--data_dir', type=str, default='Sub_data', help='Path to 10-class ImageNet subset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--save_dir', type=str, default='sava_model/', help='Path to save trained model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
parser.add_argument('--save_gradients', type=bool, default=False, help='Whether to save gradients')
parser.add_argument('--grad_dir', type=str, default='Save_grad/', help='Path to save gradients')
parser.add_argument('--sava_init_weights', type=str, default='init_weight/', help='Paht to save initial weights')
parser.add_argument('--save_concept_vector', type=str, default='concept_vector/', help='Whether to save concept vector')
parser.add_argument('--concept_vector_dir', type=str, default='concept_vector/', help='Path to save concept vector')
parser.add_argument('--model_path', type=str, default='/mnt/data/code/sava_model/VGG16_20250706_214217.pth', help='Path to the model architecture')

args = parser.parse_args()
