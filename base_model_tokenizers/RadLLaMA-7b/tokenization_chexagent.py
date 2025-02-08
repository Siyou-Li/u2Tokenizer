import os
import random
import unicodedata
from shutil import copyfile
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Callable, Optional

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import requests
import sentencepiece as spm
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from transformers import PreTrainedTokenizer, AddedToken
from transformers.convert_slow_tokenizer import import_protobuf
from transformers.utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-tokenizer": 2048,
}
SPIECE_UNDERLINE = "▁"

IMG_TOKEN_SPAN = 256

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def _list_find(
        input_list: List[Any],
        candidates: Tuple[Any],
        start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1


def _replace_closed_tag(
        input_tokens: List[Any],
        start_tags: Union[Any, Tuple[Any]],
        end_tags: Union[Any, Tuple[Any]],
        inclusive_replace_func: Callable,
        exclusive_replace_func: Callable = lambda x: x,
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end: start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed image token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start: end + 1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens


class CheXagentTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token=None,
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            use_default_system_prompt=False,
            spaces_between_special_tokens=False,
            legacy=None,
            # errors="replace",
            # image_start_tag='<|img|>',
            # image_end_tag='<|/img|>',
            # image_pad_tag='<im_patch>',
            # ref_start_tag='<|ref|>',
            # ref_end_tag='<|/ref|>',
            # box_start_tag='<bx_start>',
            # box_end_tag='<bx_end>',
            # quad_start_tag='<|quad|>',
            # quad_end_tag='<|/quad|>',
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            **kwargs,
        )
        # self.errors = errors  # how to handle errors in decoding
        # self.image_start_tag = image_start_tag
        # self.image_end_tag = image_end_tag
        # self.image_pad_tag = image_pad_tag
        # self.ref_start_tag = ref_start_tag
        # self.ref_end_tag = ref_end_tag
        # self.box_start_tag = box_start_tag
        # self.box_end_tag = box_end_tag
        # self.quad_start_tag = quad_start_tag
        # self.quad_end_tag = quad_end_tag
        # self.IMAGE_ST = (
        #     image_start_tag, image_end_tag, image_pad_tag,
        #     ref_start_tag, ref_end_tag, box_start_tag, box_end_tag,
        #     quad_start_tag, quad_end_tag,
        # )
        
        # self.img_start_id = self.convert_tokens_to_ids(self.image_start_tag)
        # self.img_end_id = self.convert_tokens_to_ids(self.image_end_tag)
        # self.img_pad_id = self.convert_tokens_to_ids(self.image_pad_tag)
        # self.ref_start_id = self.convert_tokens_to_ids(self.ref_start_tag)
        # self.ref_end_id = self.convert_tokens_to_ids(self.ref_end_tag)
        # self.box_start_id = self.convert_tokens_to_ids(self.box_start_tag)
        # self.box_end_id = self.convert_tokens_to_ids(self.box_end_tag)
        # self.quad_start_id = self.convert_tokens_to_ids(self.quad_start_tag)
        # self.quad_end_id = self.convert_tokens_to_ids(self.quad_end_tag)
        self.chat_template = DEFAULT_CHAT_TEMPLATE

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """

        def _encode_imgurl(img_tokens):
            assert img_tokens[0] == self.image_start_tag and img_tokens[-1] == self.image_end_tag
            img_tokens = img_tokens[1:-1]
            img_url = ''.join(img_tokens)
            out_img_tokens = list(img_url)
            if len(out_img_tokens) > IMG_TOKEN_SPAN:
                raise ValueError("The content in {}..{} is too long".format(self.image_start_tag, self.image_end_tag))
            out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(out_img_tokens)))
            out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.image_end_tag]
            return out_img_tokens

        if self.legacy or len(text) == 0:
            tokens = super().tokenize(text, **kwargs)
            tokens = _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)
            return tokens

        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)

    def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            errors: str = None,
            **kwargs,
    ) -> str:
        # def _decode_imgurl(img_token_ids):
        #     assert img_token_ids[0] == self.img_start_id and img_token_ids[-1] == self.img_end_id
        #     img_token_ids = img_token_ids[1:-1]
        #     img_token_ids = img_token_ids[: img_token_ids.index(self.img_pad_id)]
        #     return [self.img_start_id] + img_token_ids + [self.img_end_id]

        # token_ids = _replace_closed_tag(token_ids, self.img_start_id, self.img_end_id, _decode_imgurl)
        return super()._decode(token_ids, errors=errors or self.errors)

    def to_list_format(self, text: str):
        text = unicodedata.normalize("NFC", text)
        token_ids = self.encode(text)[1:]

        def _encode_vl_info(tokens):
            if len(tokens) == 0:
                return []
            if tokens[0] == self.img_start_id and tokens[-1] == self.img_end_id:
                key = 'image'
                tokens = tokens[: tokens.index(self.img_pad_id)]
            elif tokens[0] == self.ref_start_id and tokens[-1] == self.ref_end_id:
                key = 'ref'
            elif tokens[0] == self.box_start_id and tokens[-1] == self.box_end_id:
                key = 'box'
            elif tokens[0] == self.quad_start_id and tokens[-1] == self.quad_end_id:
                key = 'quad'
            else:
                key = 'text'
                return [{key: self.decode(tokens)}]
            return [{key: self.decode(tokens[1:-1])}]

        return _replace_closed_tag(
            token_ids,
            (self.img_start_id, self.ref_start_id, self.box_start_id, self.quad_start_id),
            (self.img_end_id, self.ref_end_id, self.box_end_id, self.quad_end_id),
            _encode_vl_info,
            _encode_vl_info,
        )

    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}:'
                text += self.image_start_tag + ele['image'] + self.image_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

    def _fetch_latest_picture(self, response, history):
        if history is None:
            history = []
        _history = history + [(response, None)]
        for q, r in _history[::-1]:
            for ele in self.to_list_format(q)[::-1]:
                if 'image' in ele:
                    return ele['image']
        return None

    def _fetch_all_box_with_ref(self, text):
        list_format = self.to_list_format(text)
        output = []
        for i, ele in enumerate(list_format):
            if 'box' in ele:
                bbox = tuple(map(int, ele['box'].replace('(', '').replace(')', '').split(',')))
                assert len(bbox) == 4
                output.append({'box': bbox})
                if i > 0 and 'ref' in list_format[i - 1]:
                    output[-1]['ref'] = list_format[i - 1]['ref'].strip()
        return output

    def draw_bbox_on_latest_picture(
            self,
            response,
            history=None,
    ) -> Optional[Image.Image]:
        image = self._fetch_latest_picture(response, history)
        if image is None:
            return None
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            h, w = image.height, image.width
        else:
            image = np.asarray(Image.open(image).convert("RGB"))
            h, w = image.shape[0], image.shape[1]
        visualizer = Visualizer(image)

        boxes = self._fetch_all_box_with_ref(response)
        if not boxes:
            return None
        color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])  # init color
        for box in boxes:
            if 'ref' in box:  # random new color for new refexps
                color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
            visualizer.draw_box((x1, y1, x2, y2), alpha=1, edge_color=color)
            if 'ref' in box:
                visualizer.draw_text(box['ref'], (x1, y1), color=color, horizontal_alignment="left")
        return visualizer.output

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
                bos_token_id
                + ([0] * len(token_ids_0))
                + eos_token_id
                + bos_token_id
                + ([0] * len(token_ids_1))
                + eos_token_id
        )

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output


class VisImage:
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    def __init__(self, img_rgb, metadata=None, scale=1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 14
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 30, 15 // scale
        )

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0,
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def get_output(self):
        return self.output
