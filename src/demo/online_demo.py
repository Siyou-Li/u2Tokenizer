import argparse
import os
import re
import sys
import bleach
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import nibabel as nib
import numpy as np
from PIL import Image
from monai.transforms import Resize
from src.model.language_model import *
import torch.nn.functional as F
import SimpleITK as sitk
import cv2

def parse_args(args):
    parser = argparse.ArgumentParser(description="M3D-LaMed chat")
    parser.add_argument('--model_name_or_path', type=str, default="/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/LaMed/output/example", choices=[])
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda:4", choices=["cuda", "cpu"])

    parser.add_argument('--seg_enable', type=bool, default=True)
    parser.add_argument('--proj_out_num', type=int, default=256)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def image_process(file_path, output_size=(32, 256, 256),mode='trilinear'):
    if file_path.endswith('.nii.gz'):
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
        img_data = torch.from_numpy(img_array)
        img_data = img_data.permute(2, 0, 1)  # shape: (D, H, W)
        new_D, new_H, new_W = output_size
        warp_img = img_data.unsqueeze(0).unsqueeze(0) 
        warp_img = F.interpolate(warp_img, size=(new_D, new_H, new_W), mode=mode, align_corners=True)
        warp_img = warp_img.squeeze(0).squeeze(0)
        img_array = warp_img.numpy()
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        # img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        # minval,maxval = np.min(img_array),np.max(img_array)
        # img_array = ((img_array-minval)/(maxval-minval)).clip(0,1)*255
        # img_array = img_array.astype(np.uint8)
        print(np.shape(img_array))
    elif file_path.endswith(('.png', '.jpg', '.bmp')):
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        img_array = img_array[np.newaxis, :, :]
    elif file_path.endswith('.npy'):
        img_array = np.load(file_path)
    else:
        raise ValueError("Unsupported file type")

    resize = Resize(spatial_size=(32, 256, 256), mode="bilinear")
    img_meta = resize(img_array)
    img_array, img_affine = img_meta.array, img_meta.affine
    print(np.shape(img_array))
    return img_array, None

args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
device = torch.device(args.device)

dtype = torch.float32
if args.precision == "bf16":
    dtype = torch.bfloat16
elif args.precision == "fp16":
    dtype = torch.half

kwargs = {"torch_dtype": dtype}
if args.load_in_4bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    )
elif args.load_in_8bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        }
    )


tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    model_max_length=args.max_length,
    padding_side="left",
    use_fast=False,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map=4,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    **kwargs
)
model = model.to(device=device)

model.eval()
print(model.__dict__)
# Gradio
examples = [
    [
        "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_00.nii.gz",
        "Please generate a medical report based on this image.",
    ],
    [
        "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_01.nii.gz",
        "What is the abnormality type in this image?",
    ],
    [
        "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_02.nii.gz",
        "What is the information of the CT image?",
    ],
    # [
    #     "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_03.npy",
    #     "Where is liver in this image? Please output the box.",
    # ],
    # [
    #     "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_04.npy",
    #     "Can you segment the lung in this image? Please output the mask.",
    # ],
    # [
    #     "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/Data/data/examples/example_05.npy",
    #     "Can you find the organs related to the balance of water and salt? Please output the box.",
    # ],
]

description = """
"""

title_markdown = ("""
# 3D Medical Image Analysis with Multi-Modal Large Language Models
""")


def extract_box_from_text(text):
    match = re.search(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    if match:
        box_coordinates = [float(coord) for coord in match.groups()]
        return box_coordinates
    else:
        return None

## to be implemented
def inference(nifti_file, input_id, temperature=1.0, top_p=0.9):

    nifti_img = nib.load(nifti_file)
    img_array = nifti_img.get_fdata()
    img_data = torch.from_numpy(img_array)
    img_data = img_data.permute(2, 0, 1)  # shape: (D, H, W)
    warp_img = img_data.unsqueeze(0).unsqueeze(0) 
    warp_img = F.interpolate(warp_img, size=(32, 256, 256), mode='trilinear', align_corners=True)
    warp_img = warp_img.squeeze(0).squeeze(0)

    raw_question = "What is the finding in chest of this image?"
    image_tokens = "<im_patch>" * 256
    question = image_tokens + ' ' + raw_question

    input_id = tokenizer(input_id, return_tensors="pt")['input_ids'].to("cuda")
    image_pt = torch.from_numpy(input_image).to("cuda")

    generation = model.generate(image_pt, input_id, max_new_tokens=512,
                                        do_sample=True, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return output_str, None


def select_slice(selected_slice):
    min_s = min(vis_box[0], vis_box[3])
    max_s = max(vis_box[0], vis_box[3])

    if min_s <= selected_slice <= max_s:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((vis_box[2],vis_box[1], vis_box[5],vis_box[4]), 'target_box')])
    else:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((0,0,0,0), 'target_box')])


def load_image(load_image):
    global image_np
    global image_rgb
    global vis_box
    global seg_mask
    vis_box = [0, 0, 0, 0, 0, 0]
    seg_mask = np.zeros((32, 256, 256), dtype=np.uint8)

    image_np, image_affine = image_process(load_image)
    image_rgb = (np.stack((image_np[0],) * 3, axis=-1) * 255).astype(np.uint8)
    # image_rgb = []
    # print(image_np[16][100])
    # for i in range(32):
    #     image_rgb.append(cv2.cvtColor(np.expand_dims(image_np[i].astype(np.uint8), axis=0), cv2.COLOR_GRAY2RGB))
    # print(np.shape(np.stack(image_rgb)))
    return (image_rgb[0], [((0,0,0,0), 'target_box')])


with gr.Blocks() as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image = gr.File(type="filepath", label="Input File")
            text = gr.Textbox(lines=1, placeholder=None, label="Text Instruction")
            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                        label="Temperature", )
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
            with gr.Row():
                btn_c = gr.ClearButton([image, text])
                btn = gr.Button("Run")
            text_out = gr.Textbox(lines=1, placeholder=None, label="Text Output")
        with gr.Column():
            image_out = gr.AnnotatedImage()
            slice_slider = gr.Slider(minimum=0, maximum=31, step=1, interactive=True, scale=1, label="Selected Slice")

    gr.Examples(examples=examples, inputs=[image, text])

    image.change(fn=load_image, inputs=[image], outputs=[image_out])
    btn.click(fn=inference, inputs=[image, text, temperature, top_p], outputs=[text_out, image_out])
    slice_slider.change(fn=select_slice, inputs=slice_slider, outputs=[image_out])
    btn_c.click()

demo.queue()
demo.launch(share=True)


