from style_template import styles

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)
def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

DEFAULT_STYLE_NAME = "(No style)"

general_prompt = "Mickey Mouse, wearing red shorts and yellow shoes"
negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
prompt_array = [
    "Bugs Bunny and Mickey stand excitedly at the entrance of Disneyland.",
    "They ride a rollercoaster, Bugs throws his carrot in the air.",
    "Bugs gets stuck inside a teacup ride, dizzy-eyed.",
    "Mickey pulls Bugs out of a popcorn machine, both covered in popcorn.",
    "They meet a crowd of fans asking for photos.",
    "The two friends watch the fireworks, smiling under the night sky."
]
id_length = 4
style_name = "Disney comic"
prompts = [general_prompt+","+prompt for prompt in prompt_array]
id_prompts = prompts[:id_length]
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
debug = 1