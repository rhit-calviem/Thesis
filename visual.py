import sys
import torch
from config import *
from models import OmniSR
from utils_data import get_test_dataset
from utils import visualize_and_save_result

def main(source=None):
    # Load model, dataset or image, visualize and save result
    model = OmniSR(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    if source is None:
        print("No input specified, using random sample from Set5.")
        dataset = get_test_dataset("Set5")
        visualize_and_save_result(model, dataset, DEVICE, save_path=VISUALIZATION_SAVE_PATH)
    elif source in TEST_DATASETS:
        print(f"Using dataset {source}")
        dataset = get_test_dataset(source)
        visualize_and_save_result(model, dataset, DEVICE, save_path=VISUALIZATION_SAVE_PATH)
    else:
        print(f"Using custom image: {source}")
        visualize_and_save_result(model, source, DEVICE, save_path=VISUALIZATION_SAVE_PATH)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
    print("Visualization saved to", VISUALIZATION_SAVE_PATH)
