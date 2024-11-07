# Improving Prompting with Knowledge Graph

This repository include the final project of Kepco KDN - GenAI 2024 Course.

Author Nguyễn Phương Anh - 22028332

I want to give special thanks to:

- All lecturers of the course - for interesting and wholesome lectures, along with many new knowledge upon the topic of AI, especially Generative AI, throught the past 6 weeks.

- Kepco KDN Group - for sponsoring, giving us free access to this useful course.

- The authors of "COMET Knowledge Graph" - for their amazing work, which is the fundamental for this project.

Introduction of this project can be found at: <https://drive.google.com/file/d/1U9rS3-fzsV3v7vEORqa2hXALpFNQgaZg/view?usp=sharing>

## Requirements

Please install the following library:

- tensorflow
- ftfy==5.1
- tensorboardX
- tqdm
- pandas
- ipython
- spacy
- transformers

For the set-up of COMET, please refer to the `README.md` inside `/comet`

## Describe

In this project, I expand the prompt to AI by adding exploited context from the prompt using COMET knowledge graph

## Test

To test the project, please run `demo.sh`. Specific code of the project is in `/comet/scripts/interactive/prompt.py` or run on Colab at <https://colab.research.google.com/drive/10iHGi-xBWbMoQ1HYNDZWsrWpMRiBmWmG?usp=sharing>
