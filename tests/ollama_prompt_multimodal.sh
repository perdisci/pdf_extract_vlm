#!/bin/bash

IMG_PATH=$1

# 1. Encode the image (removes newlines for JSON compatibility)
IMG_B64=$(base64 -w 0 ${IMG_PATH})

# 2. Send the request
curl http://localhost:11434/api/chat -d '{
"model": "gemma3:27b-it-fp16",
"messages": [
{
"role": "user",
"content": "What is represented in this image? Explain the context and content in detail. If there is any text, also extract the text, and translate it if it is not in English.",
"images": ["'$IMG_B64'"]
}
],
"stream": false
}'
