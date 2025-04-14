"""
This file defines the data decoding logic (i.e. from web tar hosted on s3 to samples feedable to packing)
"""
import io
import json
import logging
import random
import math

import os

import tarfile
from io import BytesIO
import requests 

import PIL.Image
import webdataset as wds
from webdataset.tariterators import group_by_keys, tar_file_expander, url_opener

from m4.training.enums import DatasetTypes


meta_prefix = "__"
meta_suffix = "__"

logger = logging.getLogger(__name__)
trace = False

def paths_to_wds_commands(paths: str, token: str=None):
    """Convert a list of paths to webdataset commands."""
    if token is None:
        return [f"pipe:curl -s -L {path}" for path in paths]
    else:
        return [f"pipe:curl -s -L {path} -H 'Authorization:Bearer {token}'" for path in paths]

def check_webdataset_command(command):
    if "s3:/" not in command:
        return True

    command = command.strip()
    if not command.startswith("pipe:bash"):
        return False

    if not command.endswith(".tar"):
        return False

    if "get_file.sh" not in command:
        return False

    return True


def webdoc_valid_sample(sample):
    """Check whether a sample is valid.

    :param sample: sample to be checked
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
        and sample_has_all_files(sample)
    )


def sample_has_all_files(current_sample):
    meta = current_sample.get("metadata.value", None)
    if meta is None:
        return False
    meta = meta.decode("utf-8")
    if len(meta) == 0:
        return False
    target_file_list = meta.split("\n")
    fname_keys = [key for key in current_sample.keys() if key.endswith(".fname")]
    fnames = [current_sample[key] for key in fname_keys]
    check = all([fname in fnames for fname in target_file_list])
    if not check:
        return False
    return True


class ImageDecoder:
    def __call__(self, bytes_):
        img = PIL.Image.open(io.BytesIO(bytes_))
        img.load()
        return img


# Taken from https://github.com/mlfoundations/open_clip/blob/c48111dacac55db24878af229d8a5662c03e6f1c/src/training/data.py#L180-L183
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def collate_dicts(samples):
    keys = samples[0].keys()
    batched_samples = {key: [sample[key] for sample in samples] for key in keys}
    return batched_samples


# Web Documents Utils
# Adapt group_by_keys to our webdocument format in which each samples contains several text and image files
# https://github.com/webdataset/webdataset/blob/039d74319ae55e5696dcef89829be9671802cf70/webdataset/tariterators.py#L195-L250
def group_by_keys_interleaved(data, handler=log_and_continue):
    """Return function over iterator that groups key, value pairs into samples."""
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            fname = fname.strip("./")
            if fname.endswith(".metadata.txt"):
                prefix, data_type, extension = fname.split(".")
                suffix = data_type
            else:
                prefix, idx, data_type, extension = fname.split(".")
                if data_type not in ["text", "image"]:
                    raise ValueError(f"{fname}: unknown data type {data_type}")
                suffix = idx
            if trace:
                print(
                    f"prefix: {prefix}, idx: {idx}, data_type: {data_type}, extension: {extension}, keys:"
                    f" {current_sample.keys() if isinstance(current_sample, dict) else None}"
                )
            if prefix is None:
                continue
            if current_sample is None or prefix != current_sample["__key__"]:
                valid = webdoc_valid_sample(current_sample)
                if valid:
                    yield current_sample
                elif current_sample is not None:
                    logging.warning(f"{fname}: invalid sample {current_sample} ignored")
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                raise ValueError(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
            current_sample[f"{suffix}.value"] = value
            current_sample[f"{suffix}.type"] = data_type
            current_sample[f"{suffix}.fname"] = fname
        except Exception as exn:
            exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break

    if webdoc_valid_sample(current_sample):
        yield current_sample


def _tarfile_to_webdocument_samples(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_interleaved(files, handler=handler)
    return samples


tarfile_to_webdocument_samples = wds.filters.pipelinefilter(_tarfile_to_webdocument_samples)


def _collate_texts_and_images_webdocument(data, handler=log_and_continue):
    for sample in data:
        try:
            max_example_indices = max(
                [int(key.split(".")[0]) for key in sample.keys() if key.endswith(".value") and key != "metadata.value"]
            )
            texts = [None for _ in range(max_example_indices + 1)]
            images = [None for _ in range(max_example_indices + 1)]
            for idx in range(max_example_indices + 1):
                if f"{idx}.value" not in sample:
                    continue
                if "text" in sample[f"{idx}.type"]:
                    texts[idx] = sample[f"{idx}.value"]
                elif "image" in sample[f"{idx}.type"]:
                    images[idx] = sample[f"{idx}.value"]
                else:
                    raise ValueError(f"Unknown data type: {sample[f'{idx}.type']}")
            example = {"__key__": sample["__key__"], "__url__": sample["__url__"], "texts": texts, "images": images}
            yield example
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


collate_texts_and_images_webdocument = wds.filters.pipelinefilter(_collate_texts_and_images_webdocument)


def _decode_image_and_text_webdocument(data, handler=log_and_continue):
    image_decoder = ImageDecoder()
    for sample in data:
        try:
            sample["images"] = [image_decoder(image) if image is not None else None for image in sample["images"]]
            sample["texts"] = [text.decode("utf-8") if text is not None else None for text in sample["texts"]]
            yield sample
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


decode_image_and_text_webdocument = wds.filters.pipelinefilter(_decode_image_and_text_webdocument)


# Image text pairs utils
def split_keep_2(x):
    x = x.strip("./")
    x_splitter = x.split(".")
    return x_splitter[0], x_splitter[1]


def _tarfile_to_pair_samples(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, keys=split_keep_2, handler=handler)
    return samples


tarfile_to_pair_samples = wds.filters.pipelinefilter(_tarfile_to_pair_samples)


def _decode_image_and_text_pairs(data, handler=log_and_continue):
    image_decoder = ImageDecoder()
    for sample in data:
        try:
            sample["image"] = image_decoder(sample["image"])
            sample["text"] = sample["text"].decode("utf-8")
            yield sample
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


decode_image_and_text_pairs = wds.filters.pipelinefilter(_decode_image_and_text_pairs)


# OCR utils
# Copy pasted and adapted from https://github.com/huggingface/chug/blob/cfb16882e1058b37871b61fe8f76830cef3d8750/src/chug/webdataset/doc_anno_pipe.py#L89C4-L89C4
def _decode_pdf_pages(
    sample,
    num_anno_pages,
):
    import fitz
    image_fmt = "L"
    with io.BytesIO(sample["pdf"]) as b:
        # FIXME test and use an alternate pdf reader/render as default
        assert fitz is not None, "fitz (pymupdf) is not installed and enabled"
        doc = fitz.Document(stream=b)
        num_image_pages = doc.page_count
        if num_image_pages != num_anno_pages:
            logger.warning(
                f"Mismatch between num image and num annotation pages {num_image_pages} != {num_anno_pages}"
                f" for sample {sample['__url__']}, {sample['__key__']}."
            )
        decoded_image_pages = []
        for page_index in range(num_anno_pages):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(dpi=150)
            page_image = PIL.Image.frombuffer("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            page_image = page_image.convert(image_fmt)

            decoded_image_pages += [page_image]

        return decoded_image_pages


def _decode_ocr_documents(data, handler=log_and_continue):
    # Only used for rendered_text dataset
    image_decoder = ImageDecoder()
    # Limit the number of pages to avoid OOM
    MAX_NUM_PAGES = 1
    for sample in data:
        try:
            sample_dict = json.loads(sample["json"].decode("utf-8"))
            document = dict()
            # This case is for the rendered_text dataset
            if "png" in sample:
                document["images"] = [image_decoder(sample["png"])]
                # For the rendered text dataset, we know for sure a separation is a new line.
                document["texts"] = ["\n".join(sample_dict["ocr_annotation"]["text"])]
            else:
                document["images"] = _decode_pdf_pages(sample, num_anno_pages=len(sample_dict["pages"]))
                # pdfa has a lines/words subdict, but not idl
                if "lines" in sample_dict["pages"][0]:
                    document["texts"] = [" ".join(page["lines"]["text"]) for page in sample_dict["pages"]]
                else:
                    document["texts"] = [" ".join(page["text"]) for page in sample_dict["pages"]]

            # Make sure we don't yield documents with too many pages
            if len(document["texts"]) > MAX_NUM_PAGES:
                chunked_document = dict()
                for chunk_start_index in range(0, len(document["texts"]), MAX_NUM_PAGES):
                    chunked_document["texts"] = document["texts"][
                        chunk_start_index : chunk_start_index + MAX_NUM_PAGES
                    ]
                    chunked_document["images"] = document["images"][
                        chunk_start_index : chunk_start_index + MAX_NUM_PAGES
                    ]
                    yield chunked_document
            else:
                yield document
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


decode_ocr_documents = wds.filters.pipelinefilter(_decode_ocr_documents)


# Image/question/asnwer triplets utils
def _decode_iqa_triplets(data, handler=log_and_continue):
    image_decoder = ImageDecoder()
    for sample in data:
        try:
            sample["image"] = image_decoder(sample["image"])
            sample["question"] = sample["question"].decode("utf-8")
            sample["answer"] = sample["answer"].decode("utf-8")
            yield sample
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


decode_iqa_triplets = wds.filters.pipelinefilter(_decode_iqa_triplets)


# SFT utils
def _decode_image_and_text_sft(data, handler=log_and_continue):
    image_decoder = ImageDecoder()
    for sample in data:
        try:
            sample["images"] = [image_decoder(image) for image in sample["images"] if image is not None]
            sample["texts"] = [json.loads(text.decode("utf-8")) for text in sample["texts"] if text is not None]
            yield sample
        except Exception as exn:
            exn.args = exn.args + (sample.get("stream"), sample.get("url"))
            if handler(exn):
                continue
            else:
                break


decode_image_and_text_sft = wds.filters.pipelinefilter(_decode_image_and_text_sft)


# General
def _get_web_dataset(
    urls,
    tar_to_samples_ops,
    decode_and_batch_ops,
    shuffle_initial_urls_list=False,
    shuffle_before_split_by_node_buffer_size=100,
    shuffle_before_split_by_worker_buffer_size=100,
    shuffle_after_tarfile_to_samples_buffer_size=100,
    shuffle_after_batching_buffer_size=1000,
):
    # Optionally shuffle the URLs
    if shuffle_initial_urls_list:
        random.shuffle(urls)

    # urls to command
    urls = paths_to_wds_commands(urls, token=os.getenv("HF_TOKEN"))

    pipeline_list = [wds.SimpleShardList(urls)]

    if shuffle_before_split_by_node_buffer_size is not None:
        pipeline_list.append(wds.shuffle(shuffle_before_split_by_node_buffer_size))

    pipeline_list.append(wds.split_by_node)

    if shuffle_before_split_by_worker_buffer_size is not None:
        pipeline_list.append(wds.shuffle(shuffle_before_split_by_worker_buffer_size))

    # Main processing pipeline
    pipeline_list.extend(tar_to_samples_ops)

    if shuffle_after_tarfile_to_samples_buffer_size is not None:
        pipeline_list.append(wds.shuffle(shuffle_after_tarfile_to_samples_buffer_size))

    pipeline_list.extend(decode_and_batch_ops)

    if shuffle_after_batching_buffer_size is not None:
        pipeline_list.append(wds.shuffle(shuffle_after_batching_buffer_size))

    return wds.DataPipeline(*pipeline_list)


def get_webdataset(
    urls,
    ds_type: DatasetTypes,
    batch_size: int,
    shuffle_initial_urls_list,
    shuffle_before_split_by_node_buffer_size,
    shuffle_before_split_by_worker_buffer_size,
    shuffle_after_tarfile_to_samples_buffer_size,
    shuffle_after_batching_buffer_size,
):
    if ds_type == DatasetTypes.WEB_DOCUMENTS:
        tar_to_samples_ops = [
            wds.split_by_worker,
            tarfile_to_webdocument_samples(),
        ]
        decode_and_batch_ops = [
            collate_texts_and_images_webdocument(),
            decode_image_and_text_webdocument(),
            wds.batched(batch_size, collation_fn=collate_dicts, partial=True),
        ]
        return _get_web_dataset(
            urls,
            tar_to_samples_ops,
            decode_and_batch_ops,
            shuffle_initial_urls_list,
            shuffle_before_split_by_node_buffer_size,
            shuffle_before_split_by_worker_buffer_size,
            shuffle_after_tarfile_to_samples_buffer_size,
            shuffle_after_batching_buffer_size,
        )
    elif ds_type == DatasetTypes.IMAGE_CAPTION_PAIRS:
        tar_to_samples_ops = [
            wds.split_by_worker,
            tarfile_to_pair_samples(handler=log_and_continue),
        ]
        decode_and_batch_ops = [
            decode_image_and_text_pairs(),
            wds.batched(batch_size, collation_fn=collate_dicts, partial=True),
        ]
        return _get_web_dataset(
            urls,
            tar_to_samples_ops,
            decode_and_batch_ops,
            shuffle_initial_urls_list,
            shuffle_before_split_by_node_buffer_size,
            shuffle_before_split_by_worker_buffer_size,
            shuffle_after_tarfile_to_samples_buffer_size,
            shuffle_after_batching_buffer_size,
        )
    elif ds_type == DatasetTypes.OCR:
        tar_to_samples_ops = [
            wds.split_by_worker,
            tarfile_to_pair_samples(handler=log_and_continue),
        ]
        decode_and_batch_ops = [
            decode_ocr_documents(),
            wds.batched(batch_size, collation_fn=collate_dicts, partial=True),
        ]
        return _get_web_dataset(
            urls,
            tar_to_samples_ops,
            decode_and_batch_ops,
            shuffle_initial_urls_list,
            shuffle_before_split_by_node_buffer_size,
            shuffle_before_split_by_worker_buffer_size,
            shuffle_after_tarfile_to_samples_buffer_size,
            shuffle_after_batching_buffer_size,
        )
    elif (ds_type == DatasetTypes.VQAV2_TASK_FINETUNING) or (ds_type == DatasetTypes.DOCVQA):
        tar_to_samples_ops = [
            wds.split_by_worker,
            tarfile_to_pair_samples(handler=log_and_continue),
        ]
        decode_and_batch_ops = [
            decode_iqa_triplets(),
            wds.batched(batch_size, collation_fn=collate_dicts, partial=True),
        ]
        return _get_web_dataset(
            urls,
            tar_to_samples_ops,
            decode_and_batch_ops,
            shuffle_initial_urls_list,
            shuffle_before_split_by_node_buffer_size,
            shuffle_before_split_by_worker_buffer_size,
            shuffle_after_tarfile_to_samples_buffer_size,
            shuffle_after_batching_buffer_size,
        )
    elif ds_type == DatasetTypes.SFT:
        tar_to_samples_ops = [
            wds.split_by_worker,
            tarfile_to_webdocument_samples(),
        ]
        decode_and_batch_ops = [
            collate_texts_and_images_webdocument(),
            decode_image_and_text_sft(),
            wds.batched(batch_size, collation_fn=collate_dicts, partial=True),
        ]
        return _get_web_dataset(
            urls,
            tar_to_samples_ops,
            decode_and_batch_ops,
            shuffle_initial_urls_list,
            shuffle_before_split_by_node_buffer_size,
            shuffle_before_split_by_worker_buffer_size,
            shuffle_after_tarfile_to_samples_buffer_size,
            shuffle_after_batching_buffer_size,
        )
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

IMAGE_TOKEN = "<image>"
FAKE_TOKEN_AROUND_IMAGE_V2 = "<fake_token_around_image>"
FAKE_TOKEN_AROUND_IMAGE_V1 = "\n\n"
END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"

def preprocess_webdataset_sample(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    max_image_size=384,
    vision_encoder_max_image_size=384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    max_num_samples_per_document=10,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
    max_num_images_per_document=None,
    skip_ending_two_images=True,
    skip_multiple_consecutive_images=True,
):
    """
    Return a batch of samples in the format expected by the model which
    includes `input_ids`, `pixel_values`, `attention_mask`, `image_attention_mask`,
    and `next_image_attention_mask`. The `input_ids` are sampled from the document to
    ensure it has `max_seq_len` tokens otherwise, the shorter documents are packed together.
    For each document, we sample a maximum of `max_num_samples_per_document` or `max_num_samples_for_curr_document`
    (where the latter is proportional to the length of the document and inversely proportional to the length of subsequences)
    `input_ids` with sequence length `max_seq_len` from the document. This means that
    each sample sampled can have different start index. Based on the start index of sample that
    has been sampled, we also sample a maximum of `max_num_images` images from the document.
    If there are less than `max_num_images` images in the document, we pad the images with zeros.
    The start indexes are skewed towards subsequences that contain images.

    Args:
        sample (Dict): A sample object containing the document with images and text.
        tokenizer (PretrainedTokenizer): Text tokenizer to be used.
        max_seq_len (int): Maximum sequence length of the returned text tokens.
        image_transform (Callable): Transform to be applied on the images
        max_num_images (int): Maximum number of images to be sampled per sample. If less, they are padded with zeros.
        max_num_samples_per_document (int, optional): Maximum number of samples per document to be sampled. Defaults to 10.
        prefix_seed: Prefix seed sequence for "reproducible randomness" in calls to `np.random.choice`

    Returns:
        _type_: _description_
    """
    text_batch = sample["texts"]
    image_batch = sample.get("images", None)
    if image_batch is None:
        raise ValueError("`images` must be present in the sample")

    pad_token_id = tokenizer.pad_token_id

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    last_was_image = False

    fake_token_around_image_id = tokenizer.convert_tokens_to_ids(FAKE_TOKEN_AROUND_IMAGE_V2)
    MAX_NUM_IMAGE_ROWS_AND_COLUMNS = math.ceil(max_image_size / vision_encoder_max_image_size)
    MAX_NUM_IMAGES_AFTER_SPLIT = (
        MAX_NUM_IMAGE_ROWS_AND_COLUMNS**2 + 1 * (MAX_NUM_IMAGE_ROWS_AND_COLUMNS != 1)
    ) * max_num_images

    # We need to encode the \n\n with another token that we remove afterwards to prevent the tokenizer from generating
    # the prefix token.
    double_breaking_lines_token_ids = tokenizer.encode("_\n\n", add_special_tokens=False)
    double_breaking_lines_token_ids = [
        tok for tok in double_breaking_lines_token_ids if tok not in tokenizer.encode("_", add_special_tokens=False)
    ]

    all_images = []
    all_texts = []
    for raw_images, raw_texts in zip(image_batch, text_batch):
        # Filter ones that don't have either one image and one text word
        if not any(raw_images) or not any(raw_texts):
            continue

        if max_num_images_per_document:
            num_images = sum([1 if image is not None else 0 for image in raw_images])
            if num_images > max_num_images_per_document:
                continue

        if skip_ending_two_images:
            # Skipping sequences that end with a concatenation of at least two images with no text in between
            # Order of magnitude: skipping 10% of sequences
            if (len(raw_texts) >= 2) and (raw_texts[-1] is None) and (raw_texts[-2] is None):
                continue

        def has_consecutive_nones(lst, max_num_nones=3):
            count = 0
            for item in lst:
                if item is None:
                    count += 1
                    if count == max_num_nones:
                        return True
                else:
                    count = 0
            return False

        if skip_multiple_consecutive_images and has_consecutive_nones(raw_texts):
            # We skip documents with at least 3 consecutive images
            continue

        # inds_of_texts_to_split = [
        #     i
        #     for i, text in enumerate(raw_texts)
        #     if text is not None and isinstance(text, str) and "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED" in text
        # ]
        # if inds_of_texts_to_split:
        #     splitted_raw_images, splitted_raw_texts = [], []
        #     previous_i = 0
        #     for i in inds_of_texts_to_split:
        #         splitting = raw_texts[i].split("END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED")
        #         part1, part2 = splitting[0], splitting[-1]

        #         sub_doc_images = raw_images[previous_i:i] + [None]
        #         sub_doc_texts = raw_texts[previous_i:i] + [part1.strip()]
        #         if not any(sub_doc_images):  # This can happen if all images in raw_images[0:i] are all None
        #             continue

        #         splitted_raw_images.append(sub_doc_images)
        #         splitted_raw_texts.append(sub_doc_texts)

        #         if part2.strip() == "":
        #             previous_i = i + 1
        #         else:
        #             raw_texts[i] = part2.strip()
        #             previous_i = i

        #     if previous_i < len(raw_images) and any(raw_images[previous_i:]):
        #         splitted_raw_images.append(raw_images[previous_i:])
        #         splitted_raw_texts.append(raw_texts[previous_i:])

        # else:
        #     splitted_raw_images, splitted_raw_texts = [raw_images], [raw_texts]

        # # Sanity check
        # if [len(ims) for ims in splitted_raw_images] != [len(txts) for txts in splitted_raw_texts]:
        #     raise ValueError(
        #         "Number of images and texts don't match after splitting on `END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED`."
        #         " Something core went wrong during the splitting and needs to be fixed."
        #     )
        for s_r_ims, s_r_txts in zip(raw_images, raw_texts):
            images, texts = [], []
            for image, text in zip(s_r_ims, s_r_txts):
                if text is None and image is None:
                    continue
                if (text is not None) and ((FAKE_TOKEN_AROUND_IMAGE_V2 in text) or (IMAGE_TOKEN in text)):
                    continue
                if image is not None:
                    images.append(image)
                if text is not None:
                    texts.append(text)

            all_images.append(images)
            all_texts.append(texts)

    return {
        "images": all_images,
        "texts": all_texts
    }