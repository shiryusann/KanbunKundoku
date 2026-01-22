## Data
In this repository, we open source 4 datasets:
- `annotated` The classical Chinese data we collected and annotated from Web漢文大系 in our work. It contains training/test/dev sets, the selected set used to evaluate LLMs, and labels used for different subtasks. It has the following structure:
  ```
  {
      "characters": [
          {
              "character": A Chinese character of the give classical Chinese sentence,
              "okurigana": Okurigana of the character,
              "particle": Particle of the character,
              "position": Kaeriten of the character
          },
          ...
      ],
      "japanese_translation": Japanese translation of the classical Chinese sentence,
      "japanese": the classical Chinese sentence in Japanese style Chinese characters,
      "traditional_chinese": the classical Chinese sentence in traditional Chinese characters,
      "simplified_chinese": the classical Chinese sentence in simplified Chinese characters,
      "segmentation": [
          word segmentation labels for each character
      ],
      "partofspeech": [
          POS tags for each character
      ],
      "dependencyarc": [
          dependency arc labels for each character
      ],
      "dependencytype": [
          dependency type labels for each character
      ]
  }
  ```
- `annotated_poem` The dataset created by Wang et al. (2023). We annotated this dataset.
- `annotated_two_types` We also provide a simpler dataset. It corresponds to figure 1 of our paper. We provide annotations placed to the right/left of each character.
  ```
  {
      "characters": [
          {
              "character": A Chinese character of the give classical Chinese sentence,
              "right": Annotations to the right of the character,
              "left": Annotations to the left of the character
          },
          ...
      ]
  }
  ```
- `annotated_poem_two_types` The simpler annotated dataset originally created by Wang et al. (2023)

## Code
This repository contains the following code file:
- `example.ipynb` is an example of running the dataset `annotated_poem` on the pretrained model `roberta-classical-chinese-base-char`.
- `automaton.py` contains the pushdown automaton and helper functions we used for these experiments.
- `data_process.py` is the code we used for loading the datasets.
- `evaluation.py` has encapsulated evaluation metrics used in our experiments.
- `model.py` has pytorch model frameworks we constructed.
- `pos_tag.py` contains the prompt template we used to get the POS tags of classical Chinese sentences.
- `translation.py` is used to transform annotated characters into Japanese sentences.

## Reference
You can cite our work using the following Bibtex.
```latex
@misc{li2025translationannotationcomputationalstudy,
      title={Translation via Annotation: A Computational Study of Translating Classical Chinese into Japanese}, 
      author={Zilong Li and Jie Cao},
      year={2025},
      eprint={2511.05239},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.05239}, 
}
```
