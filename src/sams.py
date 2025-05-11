from pathlib import Path
import time
import logging

import torch


CUR_DIR = Path(__file__).resolve().parent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_sam_mask_generator(
    name: str
):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    model_start = time.time()

    if name == "sam":
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam_checkpoint: str = CUR_DIR.parent / "3p/sam_ckpts/sam_vit_h_4b8939.pth"
        model_type: str = "vit_h"

        # Initialize the SAM model using the registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        
        # Mask generation
        mask_start = time.time()
        logger.info("Initializing mask generator")
        mask_generator = SamAutomaticMaskGenerator(sam)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")

    elif name == "sam2":


        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_checkpoint = CUR_DIR.parent / "3p/sam2_ckpts/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        
        # Mask generation
        mask_start = time.time()
        logger.info("Initializing mask generator")
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        

    else:
        raise ValueError(f"Unknown model name: {name}")
    

    return mask_generator