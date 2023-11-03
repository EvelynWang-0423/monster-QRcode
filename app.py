import torch
import random
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def create_code(content: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=16,
        border=0,
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # find smallest image size multiple of 256 that can fit qr
    offset_min = 8 * 16
    w, h = img.size
    w = (w + 255 + offset_min) // 256 * 256
    h = (h + 255 + offset_min) // 256 * 256
    if w > 1024:
        raise gr.Error("QR code is too large, please use a shorter content")
    bg = Image.new('L', (w, h), 128)

    # align on 16px grid
    coords = ((w - img.size[0]) // 2 // 16 * 16,
              (h - img.size[1]) // 2 // 16 * 16)
    bg.paste(img, coords)
    return bg


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    seed: int = -1,
    sampler="Euler a",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qr_code_content is None or qr_code_content == "":
        raise gr.Error("QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    print("Generating QR Code from content")
    qrcode_image = create_code(qr_code_content)

    # hack due to gradio examples
    init_image = qrcode_image

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        width=qrcode_image.width,
        height=qrcode_image.height,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        num_inference_steps=40,
    )
    return out.images[0]


css = """
#result_image {
    display: flex;
    place-content: center;
    align-items: center;
}
#result_image > img {
    height: auto;
    max-width: 100%;
    width: revert;
}
"""

with gr.Blocks(css=css) as blocks:
    gr.Markdown(
        """
# QR Code Monster v1.0
## QR Code AI Art Generator

Model used: https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster

Try our more powerful v2 here: https://qrcodemonster.art!

<a href="https://huggingface.co/spaces/monster-labs/Controlnet-QRCode-Monster-V1?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
<img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for no queue on your own hardware.</p>
"""
    )

    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="QR Code Content or URL",
                info="The text you want to encode into the QR code",
                value="",
            )

            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, disfigured, low quality, blurry, nsfw",
                info="Prompt that guides the generation away from",
            )

            with gr.Accordion(
                label="Params: The generated QR Code functionality is largely influenced by the parameters detailed below",
                open=True,
            ):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.5,
                    maximum=2.5,
                    step=0.01,
                    value=1.5,
                    label="Controlnet Conditioning Scale",
                    info="""Controls the readability/creativity of the QR code.
                    High values: The generated QR code will be more readable.
                    Low values: The generated QR code will be more creative.
                    """
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=25.0,
                    step=0.25,
                    value=7,
                    label="Guidance Scale",
                    info="Controls the amount of guidance the text prompt guides the image generation"
                )
                sampler = gr.Dropdown(choices=list(
                    SAMPLER_MAP.keys()), value="Euler a", label="Sampler")
                seed = gr.Number(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    # randomize=True,
                    info="Seed for the random number generator. Set to -1 for a random seed"
                )
            with gr.Row():
                run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image", elem_id="result_image")
    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            sampler,
        ],
        outputs=[result_image],
    )

    gr.Examples(
        examples=[
            [
                "test",
                "Baroque rococo architecture, architectural photography, post apocalyptic New York, hyperrealism, [roots], hyperrealistic, octane render, cinematic, hyper detailed, 8K",
                "",
                7,
                1.6,
                2592353769,
                "Euler a",
            ],
            [
                "https://qrcodemonster.art",
                "a centered render of an ancient tree covered in bio - organic micro organisms growing in a mystical setting, cinematic, beautifully lit, by tomasz alen kopera and peter mohrbacher and craig mullins, 3d, trending on artstation, octane render, 8k",
                "",
                7,
                1.57,
                259235398,
                "Euler a",
            ],
            [
                "test",
                "3 cups of coffee with coffee beans around",
                "",
                7,
                1.95,
                1889601353,
                "Euler a",
            ],
            [
                "https://huggingface.co",
                "A top view picture of a sandy beach with a sand castle, beautiful lighting, 8k, highly detailed",
                "sky",
                7,
                1.15,
                46200,
                "Euler a",
            ],
            [
                "test",
                "A top view picture of a sandy beach, organic shapes, beautiful lighting, bumps and shadows, 8k, highly detailed",
                "sky, water, squares",
                7,
                1.25,
                46220,
                "Euler a",
            ],
        ],
        fn=inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            sampler,
        ],
        outputs=[result_image],
        cache_examples=True,
    )
    gr.Markdown(
        """
## Notes

* The generated QR codes may not always be easily readable and may require adjusting the parameters.
* The prompt affects the quality of the generated QR code.
* The scan may work better if the phone is held further away from the screen, or if the page is zoomed out.

## Parameters

- **Input Text:** The text you want to encode into the QR code
- **Prompt:** Input a prompt to guide the QR code generation process, allowing you to control the appearance and style of the generated QR codes. Some are easier than others to generate readable QR codes.
- **Controlnet Control Scale:** Raise the control scale value to increase the readability of the QR codes or lower it to make the QR codes more creative and distinctive.

The generated QR codes might not always be easily readable. It might take a few tries with different parameters to find the right balance. This often depends on the prompt, which can be more or less suitable for QR code generation.


## How to Use

1. Input your text: Pass the text you'd like to encode into the QR code as input. Bigger text means bigger codes, which are less likely to give good results (will ressemble qr codes too much).
2. Set your prompt: Choose a prompt to guide the generation process (use all the SD tricks you like: styles, adjectives...).
3. Adjust the Controlnet Control Scale: The higher the control scale, the more readable the QR code will be, while a lower control scale leads to a more creative QR code.
4. Generate multiple codes: Since not all generated codes may be readable, you'll need to create a few codes with the same parameters to determine if any adjustments are needed.
5. Test the generated QR codes: Scan the generated QR codes to make sure they are readable and meet your requirements.
"""
    )

blocks.queue(concurrency_count=1, max_size=20, api_open=False)
blocks.launch(share=bool(os.environ.get("SHARE", False)), show_api=False)
