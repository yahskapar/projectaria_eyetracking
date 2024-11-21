# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms


def preprocess_image(et_image, size=(240, 320)):
    # Extract height and width of the input image
    h, w = et_image.shape
    # Prepare an empty tensor for the final preprocessed image
    pred_image = torch.zeros((1, 2, size[0], size[1]))

    # Process left half of the image
    pred_image[0, 0, :, :] = resize_and_normalize(et_image[:, : w // 2], size, False)
    # Process right half of the image
    pred_image[0, 1, :, :] = resize_and_normalize(et_image[:, w // 2 :], size, True)

    return pred_image


def resize_and_normalize(image, size=(240, 320), should_flip=False):
    # Convert to float for normalization
    image = image.float()
    # Normalize the image to [-0.5, 0.5]
    normalized_image = (image - torch.min(image)) / (
        torch.max(image) - torch.min(image)
    ) - 0.5

    # Flip the image if required
    if should_flip:
        normalized_image = torch.fliplr(normalized_image)

    # Add a channel dimension to match the expected input for transforms.Resize
    normalized_image = normalized_image.unsqueeze(0)  # Shape: [1, H, W]

    # Define the transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(size),  # Resize to the desired size
        ]
    )

    # Apply the resize transform
    final_image = transform(normalized_image)  # Shape: [1, size[0], size[1]]

    # Squeeze out the channel dimension before returning (optional)
    return final_image.squeeze(0)
