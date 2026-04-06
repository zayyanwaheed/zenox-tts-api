\# Zenox TTS API



A fine-tuned XTTS-v2 Chinese voice synthesis API built by ZENOX.



\## Setup



1\. Clone this repo

2\. Install dependencies:

pip install -r requirements.txt



3\. Download model from Hugging Face:

\[link coming soon]



4\. Run API:

python tts\_api.py



\## Usage



POST /synthesize

\- text: Chinese text

\- language: zh-cn

\- speaker\_wav: optional voice file for cloning

\- x-api-key: your API key



\## Example

curl -X POST http://localhost:8000/synthesize \\

&#x20; -H "x-api-key: your-key" \\

&#x20; -F "text=大家好" \\

&#x20; -F "language=zh-cn"

