"""
LangChain OCR Pipeline for Missing Financial Event Values.

This module resolves images relative to a user ID and uses an OpenAI
multimodal model via LangChain to extract missing (NaN) transaction/receipt values.
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def encode_image(image_path: Union[str, Path]) -> str:
    """Encodes an image file on disk into a base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def resolve_image_for_user(
    user_id: str,
    images_csv_path: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolves the image file path relative to a given user_id using images.csv.
    """
    base_dir = Path(__file__).resolve().parent.parent

    if images_csv_path is None:
        csv_file = base_dir / "dataset" / "images.csv"
    else:
        csv_file = Path(images_csv_path)

    if images_dir is None:
        img_directory = base_dir / "dataset" / "media" / "images"
    else:
        img_directory = Path(images_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"Images mapping file not found at: {csv_file}")

    df = pd.read_csv(csv_file)
    user_matches = df[df["user_id"] == user_id]
    if user_matches.empty:
        raise ValueError(f"No image record found for user_id: '{user_id}'")

    image_id = str(user_matches.iloc[0]["image_id"]).strip()
    image_path = img_directory / f"{image_id}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found on disk at: {image_path}")

    return image_path


def create_ocr_pipeline(
    api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY"),
    model_name: Optional[str] = os.getenv("OPENROUTER_MODEL_NAME"),
):
    """
    Creates a LangChain pipeline for image OCR to extract missing values.

    Input:
        Dict with 'image_path' (preferred, pre-resolved by DataStore) or
        'user_id' (fallback: resolves the image path from images.csv on disk).
    Output:
        Extracted text string used to complete missing/NaN values.
    """
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "<OPENAI_API_KEY>")
    resolved_model_name = model_name or os.environ.get("OPENAI_MODEL_NAME", "<OPENAI_MODEL_NAME>")

    llm = ChatOpenAI(
        model=resolved_model_name,
        api_key=resolved_api_key,
        temperature=0.0,
    )

    def prepare_multimodal_payload(inputs: Dict[str, Any]) -> list:
        if "user_id" in inputs and inputs["user_id"]:
            img_path = resolve_image_for_user(inputs["user_id"])
        elif "image_path" in inputs and inputs["image_path"]:
            img_path = Path(inputs["image_path"])
        else:
            raise KeyError("Pipeline input must include 'user_id' or 'image_path'.")

        b64_img = encode_image(img_path)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract the missing transaction or receipt amount from this image "
                        "to complete a missing NaN financial event value. "
                        "Return only the extracted value as plain text without commentary."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ]
        )
        return [message]

    pipeline = RunnableLambda(prepare_multimodal_payload) | llm | StrOutputParser()
    return pipeline
