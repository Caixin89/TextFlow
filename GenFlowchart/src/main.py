"""
GenFlowchart — minimal inference skeleton merged with existing SAM/text helpers.

Modes:
 - Batch: --input-json PATH [--image-dir DIR] --output PATH  (iterates all keys by default)
 - Single: --image PATH --question "..." --output PATH

Keep the function `infer_model_response(...)` as a placeholder to call your
existing image+question -> response pipeline. The SAM/text helper functions
(_preprocess, _text_extract, _text_remove, _get_word_from_SAM_bounding_box,
extract_bounding_boxes_and_text) are preserved unchanged.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm import tqdm
import pytesseract
import cv2
import numpy as np

# third-party / local model helpers
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from openai import OpenAI
from models.api_models import get_model_id

logger = logging.getLogger("genflowchart")
logging.basicConfig(level=logging.INFO, format="%(message)s")

mask_generator = None
openai_client = None


def load_SAM_mask_generator(device: str = "cpu"):
    global mask_generator
    sam_model_path = "weights/sam_vit_h_4b8939.pth"
    logger.info("Loading SAM mask generator model ... device=%s model_path=%s", device, sam_model_path)
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)


def load_openAI_client():
    global openai_client
    logger.info("Loading LLM client ...")
    openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))


# Extracting text from the image data
def _text_extract(image):
    #custom_config = r'--oem 3 --psm 11'
    text_regions = pytesseract.image_to_boxes(image)
    return text_regions


def _text_remove(image, text_regions):
    img = image.copy()
    text_area = []

    # Process each text region
    for x in text_regions:
        if not x.isdigit() and x != ' ':
            if x != '\n':
                text_area.append(x)
    for region in text_regions.splitlines():

        # Extract the coordinates from the region string
        x, y, x2, y2 = map(int, region.split()[1:5])
        matches = re.findall(r'[0-9a-zA-Z!@#$%^&*()=+{}\[\]:;<>,.?/\\`]', region[0])
        y = image.shape[0] - y
        y2 = image.shape[0] - y2
        # Remove the identified text region
        text_area.append([x, y, x2, y2])

        if x != x2 and y != y2 and len(matches) > 0:
            img[y2:y, x:x2] = cv2.medianBlur(image[y2:y, x:x2], 21)

    return img


def _preprocess(image):
    # converting image to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # converting images to binary images i.e white and black
    _, binary_image = cv2.threshold(gray, 150, 215, cv2.THRESH_BINARY)

    # resizing the images to get a bigger image
    image = cv2.resize(binary_image, None, fx=5, fy=5)

    # using kernel to go through a number of cells in an image
    kernel = np.ones((2, 2), np.uint8)

    # using morphology's dilation to remove noise in image
    image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return image


def _get_word_from_SAM_bounding_box(image_obj, box):
    word = []
    # going through each bounding box
    if len(image_obj.shape) > 2:
        width, height, color = image_obj.shape
    else:
        width, height = image_obj.shape
    for a in box:
        x, y, w, h = a
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if x > width:
            x = width - 1
        if y > width:
            y = width - 1
        if x != w and y != h:
            if (x + w) > height:
                continue
            if (y + h) > width:
                continue
            image_segment = image_obj[y:y + h, x:x + w]
            boxes_data = pytesseract.image_to_string(image_segment)
            word.append(boxes_data)
    return word


def extract_bounding_boxes_and_text(image_file):
    global mask_generator

    if isinstance(image_file, str):
        image_path = os.path.join("Patent Images", image_file)
        raw_image = cv2.imread(image_path)
    else:
        raw_image = image_file
    preprocessed_image = _preprocess(raw_image)
    text_stripped_image = _text_remove(preprocessed_image, _text_extract(preprocessed_image))
    fully_processed_image = cv2.cvtColor(text_stripped_image, cv2.COLOR_GRAY2RGB)
    sam_output = mask_generator.generate(fully_processed_image)

    # store the bounding box data obtained from SAM model and sort it according to the point coordinates
    box = [
        mask['bbox']
        for mask
        in sorted(sam_output, key=lambda x: x['point_coords'][0][1] if x['area'] > 10000 else False
                  )
    ]

    word = _get_word_from_SAM_bounding_box(preprocessed_image, box)

    return box, word


def generate_messages(box, word):
    prompt_mode = os.environ.get("GENFLOWCHART_PROMPT_MODE", "zero-shot")

    if prompt_mode == "few-shot":
        # Few-shot prompt template
        input1 = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: bounding box info obtained from sam(segment anything model):[[1191, 835, 51, 18], [850, 668, 60, 23], [583, 262, 52, 17], [514, 262, 62, 18], [1345, 835, 43, 18], [1230, 655, 49, 17], [475, 235, 54, 18], [1284, 656, 66, 16], [151, 237, 14, 16], [1286, 540, 14, 65], [84, 10, 264, 175], [84, 10, 264, 121], [85, 11, 262, 510], [84, 10, 263, 260], [0, 0, 1451, 949], [442, 192, 265, 132], [535, 238, 89, 15], [88, 189, 255, 139], [441, 407, 264, 131], [88, 407, 617, 131], [798, 407, 264, 132], [84, 407, 264, 131], [1155, 407, 275, 300], [1155, 407, 275, 132], [797, 611, 617, 132], [797, 612, 264, 132], [1165, 607, 256, 141], [1150, 805, 286, 131]]the text in each of those bounding boxes['', '', '', '', '', '', '', '', '', '', 'Customer places\\n\\nan order\\n', 'Yes\\n', '', 'Customer places\\nan order\\n\\nIs item still in Email customer and\\nstock? cancel order\\n\\nEmail customer with\\nHand off to carrier shipping confirmation\\nand tracking info\\n\\nDoes carrier\\n\\nNotify customer\\n\\ndeliver item?\\n\\nEmail customer with\\n\\nconfirmation of delivery\\nand return instructions\\n', 'Email customer and_\\ncancel order\\n', '', '', 'Print label\\n', 'Pack item Print label\\n', 'Hand off to carrier\\n', 'Pack item\\n', 'Does carrier\\ndeliver item?\\n', '<—No—\\n', '', '', 'Email customer with\\nconfirmation of delivery\\n\\nCoes return ——\\n']"
        knowledge1 = '''1.Customer places an order: When a customer initiates an order, this is the starting point of the process. 
2.Is the item still in stock?: A decision box where we check if the item is still in stock if it is pack the item else email the customer and cancel the order
3. Email customer and cancel order: If the item is no longer in stock, this step involves notifying the customer and canceling the order. 
4.Pack item: Once we pack the item , we print the label on the item.
5.Print label: Once we label is printed , we move to the next step handoff to carrier
6.Handoff to carrier: The item is handed over to the carrier for delivery 
7.Email customer with shipping confirmation and tracking info: Assuming the item is in stock, this step involves informing the customer about the shipment with tracking details. 7.Does the carrier deliver the item?: A branching point where the process checks if the carrier delivers the item, if it does, Email customer with confirmation of delivery and return instructions , else notify the customer and cancel the order.'''
        input2 = "Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data: bounding box info obtained from sam(segment anything model):[[42, 164, 17, 6], [27, 89, 32, 8], [95, 158, 14, 6], [166, 169, 17, 6], [129, 83, 4, 5], [59, 205, 10, 6], [160, 83, 19, 7], [129, 169, 33, 8], [26, 176, 28, 6], [46, 218, 10, 10], [143, 84, 3, 7], [58, 120, 14, 6], [97, 72, 11, 6], [104, 73, 4, 5], [46, 131, 10, 11], [56, 164, 3, 6], [12, 14, 3, 6], [66, 207, 3, 4], [111, 168, 11, 10], [58, 176, 17, 6], [129, 83, 17, 8], [111, 82, 10, 10], [46, 203, 10, 26], [39, 77, 23, 8], [12, 14, 22, 8], [46, 45, 10, 11], [162, 84, 4, 5], [68, 121, 4, 5], [97, 72, 5, 5], [42, 164, 4, 6], [59, 205, 5, 6], [96, 81, 25, 11], [96, 86, 14, 1], [46, 31, 10, 25], [51, 31, 0, 14], [46, 117, 10, 26], [51, 117, 0, 14], [51, 203, 0, 14], [51, 31, 0, 14], [51, 117, 0, 14], [95, 158, 5, 6], [57, 238, 19, 8], [51, 203, 0, 14], [63, 121, 9, 5], [8, 5, 86, 25], [8, 5, 86, 40], [8, 5, 86, 110], [122, 74, 64, 25], [6, 57, 89, 59], [6, 45, 105, 86], [6, 57, 180, 59], [0, 0, 191, 261], [124, 161, 64, 24], [6, 144, 89, 58], [6, 144, 89, 84], [19, 173, 78, 82], [19, 230, 64, 25], [19, 218, 64, 37]]the text in each of those bounding boxes['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '[Lamp doesnt work\\n', '', '', '', 'Lamp\\nplunged in?\\n', '', 'Replace bulb\\n', '', 'burned out?\\n\\nRepair lamp\\n', '', '']"
        knowledge2 = '''1.Lamp Doesn't Work: In this state , the lamp is not working and move on to check if lamp is plugged in
2. Lamp Plugged in?: A decision point. It checks whether the lamp is plugged in. If it is not then plugin lamp , else check if bulb burned out.
3.Bulb burned out :Check if bulb is burned out , if yes replace bulb else repair lamp'''
        query_line = f"Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:{word}bounding box info obtained from sam(segment anything model):{box}"
        return [
            {"role": "system", "content": "You describe flowcharts."},
            {"role": "user", "content": input1},
            {"role": "assistant", "content": knowledge1},
            {"role": "user", "content": input2},
            {"role": "assistant", "content": knowledge2},
            {"role": "user", "content": query_line}
        ]
    else:
        # Zero-shot prompt template
        return [
            {"role": "system", "content": f"Give a detailed and descriptive interpretation of the flowchart in the form of steps using following details pytesseract text recognition data:{word}bounding box info obtained from sam(segment anything model):{box}"},
        ],


# Keep run_inference as a small stub returning structured dict
def run_inference(image, question: str) -> Dict[str, Any]:
    global openai_client, mask_generator

    logger.info("Running inference for image %s with question: %s", image, question)
    box, word = extract_bounding_boxes_and_text(image)
    messages = generate_messages(box, word)
    model_id = get_model_id(os.environ.get("GENFLOWCHART_LLM"))
    prompt_mode = os.environ.get("GENFLOWCHART_PROMPT_MODE", "zero-shot")
    max_new_tokens = 500 if prompt_mode == "zero-shot" else 1000

    completion = openai_client.chat.completions.create(
        model=model_id,
        max_tokens=max_new_tokens,
        # temperature=temperature,
        messages=messages,
    )
    response = completion.choices[0].message.content
    return response


def save_output(output: Dict[str, Any], output_path: str):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved output to %s", output_path)


def infer_model_response(mask_generator: Optional[Any], llm_client: Optional[Any], image_path: Optional[str], question: str) -> str:
    """
    Placeholder wrapper for the actual image+question -> answer pipeline.
    Replace the body with your real SAM+LLM or pipeline code.
    """
    logger.debug("infer_model_response called image=%s question=%s", image_path, question)
    try:
        if mask_generator is not None and image_path is not None:
            # If you want to use extracted boxes/words, call extract_bounding_boxes_and_text
            try:
                boxes, words = extract_bounding_boxes_and_text(mask_generator, image_path)
                logger.debug("Extracted %d boxes and %d text regions", len(boxes), len(words))
            except Exception as e:
                logger.debug("SAM extraction failed: %s", e)
                boxes, words = [], []
            # Call the model/LLM-based inference here; for now call run_inference stub
            res = run_inference(mask_generator, image_path, question)
            return str(res.get("answer", ""))
        # Fallback single-image-less path: call run_inference stub
        res = run_inference(None, image_path, question)
        return str(res.get("answer", ""))
    except Exception as e:
        logger.exception("infer_model_response failed: %s", e)
        return "MODEL_PLACEHOLDER_ERROR"


def infer_batch(input_json: str, image_dir: str, output_path: str, device: str = "cpu") -> None:
    """
    Iterate dataset keys (all keys by default if `keys` is None) and produce output JSON with:
      { "0": {"key": "...", "question_id": "...", "question": "...", "response": "..."}, ... }
    """
    logger.info("Loading input JSON: %s", input_json)
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load optional models/clients only if requested
    mask_generator = load_SAM_mask_generator(device)
    llm_client = load_openAI_client()

    results: Dict[str, Dict[str, Any]] = {}
    idx = 0

    for doc_key in tqdm(data.keys(), total=len(data), desc="Processing flowchart VQA items"):
        doc = data.get(doc_key)
        if not doc:
            logger.warning("Key %s not found, skipping", doc_key)
            continue
        qa = doc.get("qa", {})
        if not isinstance(qa, dict) or len(qa) == 0:
            # No QA pairs for this document; skip
            logger.debug("No QA for key %s, skipping", doc_key)
            continue

        image_path = Path(image_dir) / f"{doc_key}.png" 

        for qid, qobj in qa.items():
            question_text = (qobj.get("Q") or "").strip()
            response_text = infer_model_response(mask_generator, llm_client, image_path, question_text)
            results[str(idx)] = {
                "key": doc_key,
                "question_id": str(qid),
                "question": question_text,
                "response": response_text,
            }
            idx += 1

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d results to %s", len(results), output_path)


def parse_args():
    p = argparse.ArgumentParser(description="GenFlowchart CLI (minimal)")
    p.add_argument("input-json", help="Path to FlowVQA JSON for batch inference (e.g., data/flowvqa/dev.json)")
    p.add_argument("--image-dir", help="Directory with images named by dataset key (used in batch mode)", default=".")
    p.add_argument("--output", default="output/genflowchart_inference.json", help="Output JSON path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    load_openAI_client()
    load_SAM_mask_generator(device=args.device)
    infer_batch(args.input_json, args.image_dir, args.output, device=args.device)
    return

if __name__ == "__main__":
    main()